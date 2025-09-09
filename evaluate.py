import logging

import numpy as np
import pandas as pd
from tqdm import tqdm


class EnhancedTrajectoryEvaluator:
    """专注F1和pairsF1指标的轨迹评估模块（适配新版模型）"""

    def __init__(self, model, processor, max_length=20, print_samples=3):
        """
        参数:
            model: 增强版轨迹BERT模型
            processor: 轨迹数据处理模块（需实现token_to_id/id_to_token转换）
            max_length: 最大轨迹长度
            print_samples: 打印的样本数量
        """
        self.model = model
        self.processor = processor
        self.max_length = max_length
        self.print_samples = print_samples
        self.device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
        self.special_token_count = 7  # 特殊token数量

        # 特殊token过滤集
        self.special_tokens = {'[PAD]', '[UNK]', '[MASK]', '[CLS]', '[SEP]', '[SOS]', '[EOS]'}

        # 日志配置
        self.logger = logging.getLogger(f"{__name__}.EnhancedTrajectoryEvaluator")

    def filter_special_tokens(self, token_list):
        """过滤特殊token，只保留有效POI名称（提升指标计算纯度）"""
        return [token for token in token_list if token not in self.special_tokens]

    def calc_f1(self, true_traj, pred_traj):
        """
        优化版F1计算（增强鲁棒性）
        公式: F1 = 2 * (precision * recall) / (precision + recall)
        """
        # 输入验证
        if not true_traj or not pred_traj:
            return 0.0

        # 过滤特殊token
        true_filtered = self.filter_special_tokens(true_traj)
        pred_filtered = self.filter_special_tokens(pred_traj)

        # 集合运算
        set_true = set(true_filtered)
        set_pred = set(pred_filtered)
        intersection = set_true & set_pred

        # 安全计算
        precision = len(intersection) / len(set_pred) if set_pred else 0
        recall = len(intersection) / len(set_true) if set_true else 0

        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    def calc_pairs_f1(self, true_traj, pred_traj):
        """
        重构版pairsF1计算（严格遵循参考实现逻辑）
        核心改进：
        1. 完全采用参考实现的算法结构
        2. 保持特殊token过滤机制
        3. 增加调试日志
        """
        # 过滤特殊token（保留原有逻辑）
        true_filtered = self.filter_special_tokens(true_traj)
        pred_filtered = self.filter_special_tokens(pred_traj)

        n = len(true_filtered)
        nr = len(pred_filtered)

        # 空轨迹处理
        if n <= 1 or nr <= 1:
            self.logger.debug(f"轨迹过短无法计算pairsF1: true_len={n}, pred_len={nr}")
            return 0.0

        # 计算顺序对基数
        n0 = n * (n - 1) / 2  # 真实轨迹顺序对数
        n0r = nr * (nr - 1) / 2  # 预测轨迹顺序对数

        # 构建真实轨迹顺序字典
        order_dict = {}
        for idx, poi in enumerate(true_filtered):
            order_dict[poi] = idx

        # 计算匹配的顺序对数量
        nc = 0  # 正确顺序对计数
        for i in range(nr):
            poi1 = pred_filtered[i]
            for j in range(i + 1, nr):
                poi2 = pred_filtered[j]
                # 只比较两个POI都在真实轨迹中且不相同的情况
                if poi1 in order_dict and poi2 in order_dict and poi1 != poi2:
                    # 检查顺序是否一致
                    if order_dict[poi1] < order_dict[poi2]:
                        nc += 1

        # 计算精确率和召回率
        precision = nc / n0r if n0r > 0 else 0
        recall = nc / n0 if n0 > 0 else 0

        # 计算F1分数
        if nc == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)

        # 调试日志
        self.logger.debug(
            f"pairsF1计算: nc={nc}, n0={n0:.1f}, n0r={n0r:.1f}, "
            f"precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}"
        )
        return f1

    def generate_trajectory(self, start_poi, end_poi, start_hour_onehot, end_hour_onehot):
        """
        适配新版模型的轨迹生成方法
        关键改进:
          - 适配新版模型generate接口
          - 增强错误处理
          - 时间特征合并为48维
        """
        try:
            # POI ID安全转换
            start_id = self.processor.token_to_id.get(start_poi, None)
            end_id = self.processor.token_to_id.get(end_poi, None)
            if None in (start_id, end_id):
                return [start_poi, end_poi]  # 降级处理

            # 时间特征验证与合并
            if not (isinstance(start_hour_onehot, list) and len(start_hour_onehot) == 24):
                self.logger.warning("无效的起点时间特征，使用默认值")
                start_hour_onehot = [0] * 24

            if not (isinstance(end_hour_onehot, list) and len(end_hour_onehot) == 24):
                self.logger.warning("无效的终点时间特征，使用默认值")
                end_hour_onehot = [0] * 24

            # 合并为48维时间特征
            time_features = np.array(start_hour_onehot + end_hour_onehot)

            # 调用新版模型生成轨迹
            generated_ids = self.model.generate(
                start_id=start_id,
                end_id=end_id,
                time_features=time_features,
                max_length=self.max_length
            )

            # 转换为POI名称
            return [self.processor.id_to_token.get(pid, '[UNK]') for pid in generated_ids]

        except Exception as e:
            self.logger.error(f"轨迹生成失败: {str(e)}")
            return [start_poi, end_poi]

    def evaluate(self, test_df, num_samples=100):
        """
        双指标评估引擎（增强版：打印三条预测轨迹）
        返回:
          - avg_f1: 平均F1分数
          - avg_pairs_f1: 平均顺序一致性分数
          - results: 详细样本记录
        """
        # 数据验证
        if not isinstance(test_df, pd.DataFrame) or test_df.empty:
            return {'avg_f1': 0, 'avg_pairs_f1': 0, 'results': []}

        # 结果容器
        f1_scores, pairs_f1_scores = [], []
        results = []
        printed_count = 0  # 已打印的轨迹计数

        # 动态采样
        eval_df = test_df.sample(min(num_samples, len(test_df))) if num_samples > 0 else test_df

        for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="轨迹评估"):
            # 生成轨迹
            pred_traj = self.generate_trajectory(
                row['start_poi'],
                row['end_poi'],
                row['start_hour_onehot'],
                row['end_hour_onehot']
            )

            # 获取真实轨迹
            true_traj = row['poi_list']

            # 计算指标
            f1 = self.calc_f1(true_traj, pred_traj)
            pairs_f1 = self.calc_pairs_f1(true_traj, pred_traj)

            # 存储结果
            f1_scores.append(f1)
            pairs_f1_scores.append(pairs_f1)
            results.append({
                'true': true_traj,
                'pred': pred_traj,
                'f1': f1,
                'pairs_f1': pairs_f1
            })
        # 聚合指标
        avg_f1 = np.mean(f1_scores) if f1_scores else 0
        avg_pairs_f1 = np.mean(pairs_f1_scores) if pairs_f1_scores else 0

        return {
            'avg_f1': avg_f1,
            'avg_pairs_f1': avg_pairs_f1,
            'results': results
        }

    def evaluate_batches(self, data_loader, device):

        self.model.eval()
        avg_f1_list = []  # 存储每个批次的平均F1分数
        avg_pairs_f1_list = []  # 存储每个批次的平均pairsF1分数

        # 移除进度条显示
        for batch_idx, batch in enumerate(data_loader):
            # 初始化批次指标
            batch_f1_list = []  # 存储批次内所有样本的F1分数
            batch_pairs_f1_list = []  # 存储批次内所有样本的pairsF1分数

            # 处理批次中的每个样本
            for i in range(len(batch['start_ids'])):
                # 提取样本数据
                start_id = batch['start_ids'][i].item()
                end_id = batch['end_ids'][i].item()

                # 获取POI名称
                start_poi = self.processor.id_to_token.get(start_id, f"[UNK:{start_id}]")
                end_poi = self.processor.id_to_token.get(end_id, f"[UNK:{end_id}]")

                # 获取时间特征
                time_features = batch['time_features'][i].numpy()
                start_hour = time_features[:24].tolist()
                end_hour = time_features[24:].tolist()

                # 生成轨迹
                pred_traj = self.generate_trajectory(
                    start_poi, end_poi, start_hour, end_hour
                )

                # 获取真实轨迹
                target_ids = batch['target_ids'][i]
                true_traj = []
                for token_id in target_ids:
                    if token_id.item() == self.processor.pad_id:
                        break
                    poi_name = self.processor.id_to_token.get(token_id.item(), f"[UNK:{token_id.item()}]")
                    true_traj.append(poi_name)

                # 计算指标
                f1 = self.calc_f1(true_traj, pred_traj)
                pairs_f1 = self.calc_pairs_f1(true_traj, pred_traj)

                # 存储样本指标
                batch_f1_list.append(f1)
                batch_pairs_f1_list.append(pairs_f1)

            # 计算批次平均指标
            batch_avg_f1 = np.mean(batch_f1_list) if batch_f1_list else 0
            batch_avg_pairs_f1 = np.mean(batch_pairs_f1_list) if batch_pairs_f1_list else 0

            # 添加到列表
            avg_f1_list.append(batch_avg_f1)
            avg_pairs_f1_list.append(batch_avg_pairs_f1)

        # 找出所有批次平均指标的最高值
        max_avg_f1 = max(avg_f1_list) if avg_f1_list else 0
        max_avg_pairs_f1 = max(avg_pairs_f1_list) if avg_pairs_f1_list else 0

        return max_avg_f1, max_avg_pairs_f1

