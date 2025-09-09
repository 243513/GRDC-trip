import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
import json
import logging
from pathlib import Path
from tqdm import tqdm
import math

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入自定义模块
from input_data import TrajectoryDataProcessor
from bert_model import Model
from evaluate import EnhancedTrajectoryEvaluator
from time2vec import ContrastiveDataGenerator

# ====================================
# 核心参数配置
# ====================================
CITY = "25poi"
BASE_DIR = Path("./data")
MAX_SEQ_LEN = 12
D_MODEL = 128
NUM_LAYERS = 4
NUM_HEADS = 4
DFF = 128
TIME_DIM = 48
BATCH_SIZE = 4
PRETRAIN_EPOCHS = 1
FINETUNE_EPOCHS = 1
LEARNING_RATE = 1e-3
GRAD_CLIP = 1.0
SPECIAL_TOKEN_COUNT = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 保存配置
config = {
    "city": CITY, "max_seq_len": MAX_SEQ_LEN, "d_model": D_MODEL,
    "num_layers": NUM_LAYERS, "num_heads": NUM_HEADS, "dff": DFF,
    "time_dim": TIME_DIM, "batch_size": BATCH_SIZE,
    "pretrain_epochs": PRETRAIN_EPOCHS, "finetune_epochs": FINETUNE_EPOCHS,
    "learning_rate": LEARNING_RATE, "grad_clip": GRAD_CLIP,
    "special_token_count": SPECIAL_TOKEN_COUNT,
    "device": str(DEVICE)
}


# ====================================
# PyTorch数据集类
# ====================================
class PretrainDataset(Dataset):
    """预训练数据集（MLM + 视图对比）"""

    def __init__(self, pretrain_data):
        self.input_ids = torch.tensor(pretrain_data['input_ids'], dtype=torch.long)
        self.attention_mask = torch.tensor(pretrain_data['attention_mask'], dtype=torch.long)
        self.labels = torch.tensor(pretrain_data['labels'], dtype=torch.long)
        self.time_features = torch.tensor(pretrain_data['time_features'], dtype=torch.float32)
        self.view_pair_ids = torch.tensor(pretrain_data['view_pair_id'], dtype=torch.long)

    def __len__(self): return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
            'time_features': self.time_features[idx],
            'view_pair_id': self.view_pair_ids[idx]
        }


class FinetuneDataset(Dataset):
    """微调数据集（轨迹生成）"""

    def __init__(self, finetune_set):
        """
        参数:
            finetune_set: 单个数据集字典（训练集、验证集或测试集）
        """
        # 从 condition 中提取起点和终点ID
        condition = finetune_set['condition']

        # 确保 condition 是整数元组
        self.start_ids = torch.tensor([int(c[0]) for c in condition], dtype=torch.long)
        self.end_ids = torch.tensor([int(c[1]) for c in condition], dtype=torch.long)

        self.input_ids = torch.tensor(finetune_set['input_ids'], dtype=torch.long)
        self.attention_mask = torch.tensor(finetune_set['attention_mask'], dtype=torch.long)
        self.target_ids = torch.tensor(finetune_set['target_ids'], dtype=torch.long)
        self.time_features = torch.tensor(finetune_set['time_features'], dtype=torch.float32)

    def __len__(self): return len(self.start_ids)

    def __getitem__(self, idx):
        return {
            'start_ids': self.start_ids[idx],
            'end_ids': self.end_ids[idx],
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'target_ids': self.target_ids[idx],
            'time_features': self.time_features[idx]
        }


# ====================================
# 训练阶段管理
# ====================================
def run_training_phase(model, optimizer, loader, phase, device):
    """执行单一训练阶段（改进：分离学习率）"""
    model.train()
    model.set_training_stage(phase)
    total_loss = 0.0
    phase_desc = {"pretrain": "MLM预训练", "finetune": "轨迹微调"}.get(phase)

    # 分离参数（基础参数和对比学习参数）
    base_params = []
    contrast_params = []
    for name, param in model.named_parameters():
        if "view_contrast" in name:
            contrast_params.append(param)
        else:
            base_params.append(param)

    # 创建优化器组（不同学习率）
    param_groups = [
        {"params": base_params, "lr": LEARNING_RATE},
        {"params": contrast_params, "lr": LEARNING_RATE * 10}  # 对比学习使用更高学习率
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-5)

    # 初始化详细损失记录器
    loss_components = {}
    if phase == "pretrain":
        loss_components = {'mlm_loss': 0.0, 'contrast_loss': 0.0}
    elif phase == "finetune":
        loss_components = {'gen_loss': 0.0}  # 只保留生成损失

    with tqdm(loader, desc=phase_desc, unit="batch") as pbar:
        for batch in pbar:
            # 移动数据到设备
            batch = {k: v.to(device) for k, v in batch.items()}

            # 前向传播
            outputs = model(batch, task_type=phase)
            loss = outputs['total_loss']

            # 记录详细损失
            if phase == "pretrain":
                # 安全处理损失值（可能是张量或浮点数）
                mlm_loss = outputs['mlm_loss']
                contrast_loss = outputs['contrast_loss']

                # 如果是张量则提取值，否则直接使用
                loss_components['mlm_loss'] += mlm_loss.item() if isinstance(mlm_loss, torch.Tensor) else mlm_loss
                loss_components['contrast_loss'] += contrast_loss.item() if isinstance(contrast_loss,
                                                                                       torch.Tensor) else contrast_loss
            elif phase == "finetune":
                gen_loss = outputs['gen_loss']
                loss_components['gen_loss'] += gen_loss.item() if isinstance(gen_loss, torch.Tensor) else gen_loss

            # 梯度缩放（针对对比学习）
            scaled_loss = loss * 5.0  # 放大梯度以增强对比学习

            # 反向传播
            optimizer.zero_grad()
            scaled_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)

            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    # 计算平均损失
    num_batches = len(loader)
    avg_total_loss = total_loss / num_batches

    # 计算平均详细损失
    for key in loss_components:
        loss_components[key] /= num_batches

    return avg_total_loss, loss_components


# ====================================
# 验证评估
# ====================================
def evaluate_model(model, evaluator, df, phase_name, num_samples=100):
    """评估模型并返回核心指标"""
    results = evaluator.evaluate(df, num_samples=num_samples)
    f1 = results['avg_f1']
    pairs_f1 = results['avg_pairs_f1']
    geo_score = results.get('avg_geo_score', 0.0)  # 新增地理一致性指标
    return f1, pairs_f1, geo_score, results


# ====================================
# 主函数
# ====================================
def main():
    # 创建目录
    model_dir = BASE_DIR / "models" / CITY
    result_dir = BASE_DIR / "results" / CITY
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # 保存配置
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"开始{CITY}城市轨迹模型训练，设备: {DEVICE}")
    run_start_time = time.time()

    # 设置随机种子
    seed = int(time.time() * 1000) % 1000000
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    processor = TrajectoryDataProcessor(
        city=CITY, max_seq_len=MAX_SEQ_LEN, base_dir=BASE_DIR
    )
    processor.load_data()
    processor.split_raw_data()
    processor.create_vocab()
    processor.compute_distance_matrix()
    processor.process_trajectories()
    processor.generate_transition_matrix()
    processor.load_user_history()
    processor.compute_trajectory_length_map()
    processor.augment_training_set()

    train_df = processor.train_df
    val_df = processor.val_df
    test_df = processor.test_df
    if 'is_augmented' in train_df.columns:
        train_df['is_augmented'] = train_df['is_augmented'].fillna(False)
        num_augmented = len(train_df[train_df['is_augmented']])
    else:
        num_augmented = 0
    contrast_generator = ContrastiveDataGenerator(
        city=CITY, trajectory_processor=processor, num_ns=3, base_dir=BASE_DIR
    )
    contrast_samples = contrast_generator.generate_contrastive_samples(n_samples=len(train_df))
    model = Model(
        vocab_size=len(processor.token_to_id),
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dff=DFF,
        max_len=MAX_SEQ_LEN,
        time_dim=TIME_DIM,
        distance_matrix=processor.distance_matrix,
        transition_matrix=processor.transition_matrix,
        traj_length_map=processor.traj_length_map,
        id_to_token=processor.id_to_token,
        use_relation_encoder=False,
        use_contrastive=True,
        use_spatial_constraints=True
    ).to(DEVICE)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.02)
    model.apply(init_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    pretrain_data = processor.prepare_pretrain_data()
    finetune_data = processor.prepare_finetune_data()

    pretrain_dataset = PretrainDataset(pretrain_data)

    train_finetune_dataset = FinetuneDataset(finetune_data['train'])
    val_finetune_dataset = FinetuneDataset(finetune_data['val'])
    test_finetune_dataset = FinetuneDataset(finetune_data['test'])

    num_workers = min(4, os.cpu_count() // 2)
    pretrain_loader = DataLoader(
        pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    train_finetune_loader = DataLoader(
        train_finetune_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_finetune_loader = DataLoader(
        val_finetune_dataset, batch_size=BATCH_SIZE,
        num_workers=num_workers, pin_memory=True
    )
    test_finetune_loader = DataLoader(
        test_finetune_dataset, batch_size=BATCH_SIZE,
        num_workers=num_workers, pin_memory=True
    )

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(PRETRAIN_EPOCHS):
        epoch_start = time.time()
        avg_total_loss, loss_components = run_training_phase(
            model, optimizer, pretrain_loader, "pretrain", DEVICE
        )
        mlm_loss = loss_components['mlm_loss']
        contrast_loss = loss_components['contrast_loss']
        model.eval()
        with torch.no_grad():
            val_loss = 0
            mlm_loss_total = 0
            contrast_loss_total = 0

            for batch in pretrain_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                # 前向传播
                outputs = model(batch, task_type='pretrain')
                val_loss += outputs['total_loss'].item()
                mlm_loss_val = outputs['mlm_loss']
                contrast_loss_val = outputs['contrast_loss']
                mlm_loss_total += mlm_loss_val.item() if isinstance(mlm_loss_val, torch.Tensor) else mlm_loss_val
                contrast_loss_total += contrast_loss_val.item() if isinstance(contrast_loss_val,
                                                                              torch.Tensor) else contrast_loss_val

            num_batches = len(pretrain_loader)
            val_loss /= num_batches
            mlm_loss_avg = mlm_loss_total / num_batches
            contrast_loss_avg = contrast_loss_total / num_batches

        scheduler.step(-val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 保存完整的模型状态和配置
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),  # 正确的键名
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'config': config,
                'seed': seed
            }, model_dir / "best_pretrain_model.pt")
            logger.info(f"已保存预训练模型，epoch={epoch}, val_loss={val_loss:.4f}")

        logger.info(
            f"Epoch {epoch + 1}/{PRETRAIN_EPOCHS} | "
            f"训练损失: MLM={mlm_loss:.4f} 对比={contrast_loss:.4f}  | "
            f"时间: {time.time() - epoch_start:.1f}s"
        )

    model.set_training_stage('pretrain')
    pretrain_model_path = model_dir / "best_pretrain_model.pt"
    if os.path.exists(pretrain_model_path):
        logger.info(f"找到预训练模型文件: {pretrain_model_path}")
        checkpoint = torch.load(pretrain_model_path)

        # 检查配置是否匹配
        if checkpoint.get('config', {}).get('d_model', D_MODEL) == D_MODEL:
            # 使用正确的键名 'model_state_dict'
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("成功加载预训练模型")
        else:
            logger.warning("配置不匹配，跳过加载预训练模型")
            # 重新初始化模型
            model.apply(init_weights)
    else:
        logger.warning("未找到预训练模型，使用初始权重")

    # =====================
    # 6. 微调阶段
    # =====================
    logger.info(f" 开始微调 - {FINETUNE_EPOCHS}轮")
    evaluator = EnhancedTrajectoryEvaluator(model, processor, MAX_SEQ_LEN)
    best_val_f1 = 0.0
    best_val_pairs_f1 = 0.0
    best_val_score = 0.0  # 最佳验证分数（F1 + PairsF1）

    for epoch in range(FINETUNE_EPOCHS):
        epoch_start = time.time()

        # 训练
        avg_total_loss, loss_components = run_training_phase(
            model, optimizer, train_finetune_loader, "finetune", DEVICE
        )

        # 提取详细损失
        gen_loss = loss_components['gen_loss']

        # 验证
        model.eval()
        val_f1, val_pairs_f1, _, _ = evaluate_model(
            model, evaluator, val_df, f"验证集", num_samples=len(val_df)
        )

        # 计算验证分数
        val_score = val_f1 + val_pairs_f1

        # 更新学习率
        scheduler.step(val_score)

        # 保存最佳模型
        if val_score > best_val_score:
            best_val_score = val_score
            best_val_f1 = val_f1
            best_val_pairs_f1 = val_pairs_f1
            # 保存完整的模型状态和配置
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),  # 正确的键名
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': gen_loss,
                'config': config,
                'seed': seed
            }, model_dir / "best_finetune_model.pt")
            logger.info(f"已保存微调模型，epoch={epoch}, val_score={val_score:.4f}")

        logger.info(
            f"Epoch {epoch + 1}/{FINETUNE_EPOCHS} | "
            f"训练损失: 生成={gen_loss:.4f}   "
            f"时间: {time.time() - epoch_start:.1f}s"
        )

    # 加载最佳微调模型
    model.set_training_stage('finetune')
    finetune_model_path = model_dir / "best_finetune_model.pt"
    if os.path.exists(finetune_model_path):
        checkpoint = torch.load(finetune_model_path)

        # 检查配置是否匹配
        if checkpoint.get('config', {}).get('d_model', D_MODEL) == D_MODEL:
            # 使用正确的键名 'model_state_dict'
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("成功加载微调模型")
        else:
            logger.warning("配置不匹配，跳过加载微调模型")
    else:
        logger.warning("未找到微调模型")

    # =====================
    # 7. 最终评估
    # =====================
    logger.info(" 最终评估")
    model.eval()
    # 测试集评估（分批次）
    test_max_avg_f1, test_max_avg_pairs_f1 = evaluator.evaluate_batches(test_finetune_loader, DEVICE)

    # 保存结果
    run_time = time.time() - run_start_time
    results = {
        'seed': seed,
        'test_f1': test_max_avg_f1,
        'test_pairs_f1': test_max_avg_pairs_f1,
        'run_time': run_time
    }

    logger.info(f"最终结果: F1={test_max_avg_f1:.4f}, PairsF1={test_max_avg_pairs_f1:.4f}")
    logger.info(f"总耗时: {run_time:.1f}秒")

    # 保存结果
    result_file = result_dir / "results.json"
    with open(result_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": config,
            "results": results
        }, f, indent=2)

    logger.info(f"结果已保存到: {result_file}")


if __name__ == "__main__":
    main()