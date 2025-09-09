import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import logging
logger = logging.getLogger(__name__)
class Model(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, dff,
                 max_len=12, time_dim=48, distance_matrix=None,
                 transition_matrix=None, traj_length_map=None, id_to_token=None,
                 use_relation_encoder=False, use_contrastive=True, use_spatial_constraints=True):
        super().__init__()
        # 基础参数
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.time_dim = time_dim
        self.training_stage = 'pretrain'

        self.use_relation_encoder = use_relation_encoder
        self.use_contrastive = use_contrastive
        self.use_spatial_constraints = use_spatial_constraints
        self.distance_matrix = distance_matrix
        self.transition_matrix = transition_matrix
        self.traj_length_map = traj_length_map
        self.id_to_token = id_to_token
        if self.traj_length_map:
            lengths = list(self.traj_length_map.values())
            self.global_avg_length = math.floor(sum(lengths) / len(lengths))
        else:
            self.global_avg_length = 3
        self.pad_id = 0
        self.unk_id = 1
        self.mask_id = 2
        self.cls_id = 3
        self.sep_id = 4
        self.sos_id = 5
        self.eos_id = 6
        self.special_token_count = 7
        # 新增权重参数
        self.relation_weight = 0.2
        self.transition_weight = 0.7
        self.mlm_weight = 2.0
        self.poi_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.time_embedding = nn.Linear(time_dim, d_model)
        self.gated_fusion = GatedEmbedding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dff, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        if self.use_contrastive:
            self.view_contrast = ViewContrastiveLearner(d_model)
        else:
            logger.info("对比学习模块已禁用")

        self.mlm_head = nn.Linear(d_model, vocab_size)

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, num_heads, dff, activation='gelu', batch_first=True),
            num_layers=num_layers
        )

        self.gen_head = nn.Linear(d_model, vocab_size)

        if self.use_relation_encoder:
            self.relation_encoder = RelationEncoder(d_model)
        else:
            logger.info("关系编码器已禁用")

        if self.use_contrastive and self.use_relation_encoder:
            self.rep_fusion = RepresentationFusion(d_model)
        else:
            self.rep_fusion = None

        if self.use_contrastive and self.use_relation_encoder:
            self.fusion_adapter = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.ReLU(),
                nn.LayerNorm(d_model)
            )
        else:
            self.fusion_adapter = None

        # 损失函数（使用标准MLM损失）
        self.mlm_loss = MLMLoss(vocab_size, ignore_index=-100)
        self.traj_gen_loss = TrajectoryGenerationLoss(self.pad_id)
    def set_training_stage(self, stage):
        valid_stages = ['pretrain', 'finetune']
        if stage not in valid_stages:
            raise ValueError(f"无效训练阶段: {stage}，可选: {valid_stages}")
        self.training_stage = stage
        logger.info(f"切换到训练阶段: {stage}")

    def forward(self, inputs, task_type=None):
        if task_type is None:
            task_type = self.training_stage

        if task_type == 'pretrain':
            return self.pretrain_forward(inputs)
        elif task_type == 'finetune':
            return self.finetune_forward(inputs)
        else:
            raise ValueError(f"未知任务类型: {task_type}")

    def pretrain_forward(self, inputs):
        """预训练阶段：MLM + 视图对比"""
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        time_features = inputs['time_features']
        labels = inputs['labels']
        view_pair_ids = inputs['view_pair_id']

        embeddings = self._create_embeddings(input_ids, time_features)

        encoder_output = self.encoder(embeddings, src_key_padding_mask=(attention_mask == 0))
        cls_representation = encoder_output[:, 0, :]
        mlm_logits = self.mlm_head(encoder_output)
        mlm_loss = self.mlm_loss(mlm_logits, labels, attention_mask) * self.mlm_weight
        if self.use_contrastive:
            contrast_loss = self._compute_view_contrast_loss(cls_representation, view_pair_ids)
            total_loss = mlm_loss + 0.2 * contrast_loss#权重调整
        else:
            contrast_loss = 0.0
            total_loss = mlm_loss

        return {
            'mlm_loss': mlm_loss,
            'contrast_loss': contrast_loss,
            'total_loss': total_loss,
            'logits': mlm_logits
        }

    def finetune_forward(self, inputs):
        start_ids = inputs['start_ids']
        end_ids = inputs['end_ids']
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        time_features = inputs['time_features']
        target_ids = inputs['target_ids']

        batch_size = start_ids.size(0)
        input_seq_len = input_ids.size(1)
        target_seq_len = target_ids.size(1)
        if input_seq_len != target_seq_len:
            if target_seq_len > input_seq_len:
                target_ids = target_ids[:, :input_seq_len]
            else:
                pad_len = input_seq_len - target_seq_len
                target_ids = F.pad(target_ids, (0, pad_len), value=self.pad_id)
        anchor = torch.stack([start_ids, end_ids], dim=1)
        positive = self._sample_positive(input_ids)
        negatives = self._sample_negatives(input_ids)
        anchor_emb = self.poi_embedding(anchor)  # [batch, 2, d_model]
        pos_emb = self.poi_embedding(positive.unsqueeze(1))  # [batch, 1, d_model]
        neg_emb = self.poi_embedding(negatives)  # [batch, num_ns, d_model]
        if self.use_relation_encoder:
            relation_vector = self.relation_encoder(anchor_emb, pos_emb, neg_emb)
            poi_emb_adjustment = self._adjust_poi_embeddings(relation_vector)
            self.poi_embedding.weight.data += poi_emb_adjustment * self.relation_weight
        else:
            relation_vector = torch.zeros(batch_size, self.d_model, device=anchor_emb.device)
        input_emb = self._create_embeddings(input_ids, time_features)
        encoder_output = self.encoder(input_emb, src_key_padding_mask=(attention_mask == 0))
        cls_representation = encoder_output[:, 0, :]
        if self.use_contrastive and self.use_relation_encoder and self.rep_fusion:
            fused_representation = self.rep_fusion(cls_representation, relation_vector)
        else:
            fused_representation = cls_representation
        logits = self._generate_trajectory(input_emb, encoder_output, fused_representation)
        if self.use_relation_encoder:
            self.poi_embedding.weight.data -= poi_emb_adjustment * self.relation_weight
        gen_loss = self.traj_gen_loss(logits, target_ids)

        return {
            'gen_loss': gen_loss,
            'total_loss': gen_loss,
            'logits': logits
        }

    def _adjust_poi_embeddings(self, relation_vector):
        with torch.no_grad():
            all_poi_emb = self.poi_embedding.weight.data
            similarities = F.cosine_similarity(
                relation_vector.mean(dim=0, keepdim=True),
                all_poi_emb,
                dim=-1
            )
            adjustment = similarities.unsqueeze(-1) * relation_vector.mean(dim=0)

            return adjustment

    def generate(self, start_id, end_id, time_features, max_length=12,
                 override_relation=None, override_contrastive=None, override_spatial=None):

        use_relation = override_relation if override_relation is not None else self.use_relation_encoder
        use_contrastive = override_contrastive if override_contrastive is not None else self.use_contrastive
        use_spatial = override_spatial if override_spatial is not None else self.use_spatial_constraints

        device = next(self.parameters()).device
        start_id_t = torch.tensor([start_id], device=device)
        end_id_t = torch.tensor([end_id], device=device)
        time_features_t = torch.tensor(time_features, dtype=torch.float32, device=device).unsqueeze(0)
        anchor = torch.stack([start_id_t, end_id_t], dim=1)
        positive = torch.tensor([start_id], device=device)
        negatives = torch.tensor([[start_id, end_id]], device=device)
        if use_relation:
            with torch.no_grad():
                anchor_emb = self.poi_embedding(anchor)  # [1, 2, d_model]
                pos_emb = self.poi_embedding(positive.unsqueeze(1))  # [1, 1, d_model]
                neg_emb = self.poi_embedding(negatives)  # [1, 2, d_model]
                relation_vector = self.relation_encoder(anchor_emb, pos_emb, neg_emb)

                poi_emb_adjustment = self._adjust_poi_embeddings(relation_vector)
                original_poi_emb = self.poi_embedding.weight.data.clone()
                self.poi_embedding.weight.data += poi_emb_adjustment * self.relation_weight
        seq = [self.sos_id, start_id]
        input_ids = torch.tensor([seq], device=device)

        expected_length = self._get_expected_length(start_id, end_id)

        start_poi = self.id_to_token.get(start_id, '')
        end_poi = self.id_to_token.get(end_id, '')
        if start_poi and end_poi and self.distance_matrix is not None:
            if start_poi in self.distance_matrix.index and end_poi in self.distance_matrix.columns:
                total_distance = self.distance_matrix.loc[start_poi, end_poi]
            else:
                total_distance = 0
        else:
            total_distance = 0
        current_distance = 0
        current_position = start_poi
        visited_pois = {start_id}
        for step in range(2, min(max_length, expected_length + 2)):
            # 创建输入嵌入
            input_emb = self._create_embeddings(input_ids, time_features_t)
            encoder_output = self.encoder(input_emb)
            cls_representation = encoder_output[:, 0, :]
            if use_contrastive and use_relation and self.rep_fusion:
                fused_representation = self.rep_fusion(cls_representation, relation_vector)
            else:
                fused_representation = cls_representation
            with torch.no_grad():
                logits = self._generate_trajectory(input_emb, encoder_output, fused_representation)
                next_logits = logits[0, -1, :]
                next_probs = F.softmax(next_logits, dim=-1)
                next_probs = self._apply_generation_constraints(
                    next_probs, step, max_length, expected_length, end_id,
                    visited_pois, total_distance, current_position, use_spatial
                )
                if next_probs.sum() == 0:
                    next_id = end_id
                else:
                    next_id = torch.argmax(next_probs).item()

            seq.append(next_id)
            input_ids = torch.tensor([seq], device=device)

            if next_id != end_id:
                visited_pois.add(next_id)
            next_poi = self.id_to_token.get(next_id, '')
            if next_poi and current_position and next_poi in self.distance_matrix.columns:
                current_distance += self.distance_matrix.loc[current_position, next_poi]
                current_position = next_poi
            if next_id == end_id or len(seq) >= min(max_length, expected_length + 1):
                break
        if seq[-1] != end_id:
            seq.append(end_id)
        if use_relation:
            self.poi_embedding.weight.data = original_poi_emb
        filtered_seq = [poi for poi in seq if poi >= self.special_token_count]
        return filtered_seq

    def _apply_generation_constraints(self, next_probs, step, max_length, expected_length, end_id,
                                      visited_pois, total_distance, current_position, use_spatial):
        next_probs[:self.special_token_count] = 0
        if step < min(max_length, expected_length + 1) - 1:
            next_probs[end_id] = 0

        if use_spatial:
            for poi_id in visited_pois:
                if 0 <= poi_id < self.vocab_size:
                    next_probs[poi_id] = 0
        if use_spatial and total_distance > 0 and current_position:
            # 计算剩余距离比例
            remaining_ratio = (step - 2) / (expected_length - 2)
            ideal_distance = remaining_ratio * total_distance

            # 调整候选POI的概率
            for candidate_id in range(self.special_token_count, self.vocab_size):
                candidate_poi = self.id_to_token.get(candidate_id, '')
                if candidate_poi and candidate_poi in self.distance_matrix.index:
                    # 计算当前点到候选点的距离
                    dist_to_candidate = self.distance_matrix.loc[current_position, candidate_poi]

                    dist_diff = abs(dist_to_candidate - ideal_distance)
                    adjustment = math.exp(-dist_diff / (total_distance * 0.1))
                    next_probs[candidate_id] *= adjustment
        return next_probs

    def _create_embeddings(self, input_ids, time_features):
        """创建嵌入表示（使用门控融合）"""
        batch_size, seq_len = input_ids.shape
        poi_emb = self.poi_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)

        if time_features.dim() == 2:
            time_emb = self.time_embedding(time_features).unsqueeze(1)
            time_emb = time_emb.expand(-1, seq_len, -1)
        else:
            time_emb = self.time_embedding(time_features)
        return self.gated_fusion(poi_emb, pos_emb, time_emb)

    def _compute_view_contrast_loss(self, cls_representation, view_pair_ids):
        """计算视图对比损失"""
        if view_pair_ids.dim() > 1:
            view_pair_ids = view_pair_ids.squeeze()
        view1_indices = view_pair_ids[::2]
        view2_indices = view_pair_ids[1::2]

        view1_indices = view1_indices.clamp(0, cls_representation.size(0) - 1)
        view2_indices = view2_indices.clamp(0, cls_representation.size(0) - 1)
        view1_reps = cls_representation[view1_indices]
        view2_reps = cls_representation[view2_indices]
        return self.view_contrast(view1_reps, view2_reps)
    def _sample_positive(self, input_ids):
        """从输入序列中随机采样正样本（中间点）"""
        batch_size, seq_len = input_ids.shape
        positives = []
        for i in range(batch_size):
            if seq_len > 2:
                mid_idx = random.randint(1, seq_len - 2)
                positives.append(input_ids[i, mid_idx].item())
            else:
                positives.append(input_ids[i, 0].item())

        return torch.tensor(positives, device=input_ids.device)

    def _sample_negatives(self, input_ids):
        """采样负样本（不在当前序列中的POI）"""
        batch_size, seq_len = input_ids.shape
        negatives = []
        # 有效POI范围（排除特殊token）
        valid_poi_range = range(self.special_token_count, self.vocab_size)
        all_pois = set(valid_poi_range)

        for i in range(batch_size):
            current_pois = set(input_ids[i].tolist())
            candidate_negatives = list(all_pois - current_pois)
            if len(candidate_negatives) >= 2:
                negs = random.sample(candidate_negatives, 2)
            elif candidate_negatives:
                negs = random.choices(candidate_negatives, k=2)  # 重复使用
            else:
                negs = [self.unk_id, self.unk_id]  # 回退

            negatives.append(negs)

        return torch.tensor(negatives, device=input_ids.device)

    def _get_expected_length(self, start_id, end_id):
        """获取期望轨迹长度"""
        key = (start_id, end_id)
        if key in self.traj_length_map:
            return min(self.traj_length_map[key], self.max_len)
        return min(self.global_avg_length, self.max_len)

    def _generate_trajectory(self, tgt, memory, fused_representation=None):
        """生成轨迹序列（增强转移概率影响）"""
        # 解码过程
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        if fused_representation is not None and self.fusion_adapter:
            batch_size, seq_len, _ = tgt.size()
            fused_rep_expanded = fused_representation.unsqueeze(1).expand(-1, seq_len, -1)
            tgt_with_fused = torch.cat([fused_rep_expanded, tgt], dim=-1)
            tgt_adjusted = self.fusion_adapter(tgt_with_fused)
        else:
            tgt_adjusted = tgt

        output = self.decoder(tgt_adjusted, memory, tgt_mask=tgt_mask)
        logits = self.gen_head(output)
        if self.transition_matrix is not None and not self.transition_matrix.empty:
            logits = self._fuse_transition_probs(logits, tgt)
        return logits
    def _fuse_transition_probs(self, logits, tgt):
        batch_size, seq_len, vocab_size = logits.shape
        if seq_len > 1:
            last_pois = torch.argmax(tgt[:, -1, :], dim=-1)
        else:
            last_pois = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        trans_probs = torch.zeros(batch_size, vocab_size, device=logits.device)

        for i in range(batch_size):
            poi_id = last_pois[i].item()
            if (0 <= poi_id < self.vocab_size and
                    poi_id in self.transition_matrix.index):
                trans_probs[i] = torch.tensor(
                    self.transition_matrix.loc[poi_id].values,
                    device=logits.device
                )
            else:
                trans_probs[i] = torch.ones(self.vocab_size, device=logits.device) / self.vocab_size

        trans_logits = torch.log(trans_probs + 1e-9)
        fused_logits = logits + trans_logits.unsqueeze(1) * self.transition_weight

        return fused_logits

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
class RelationEncoder(nn.Module):
    def __init__(self, d_model, num_relations=3):
        super().__init__()
        self.d_model = d_model
        self.num_relations = num_relations
        self.relation_layers = nn.ModuleList([
            nn.Linear(3 * d_model, d_model) for _ in range(num_relations)
        ])

        self.relation_attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

        self.fusion_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, anchor, positive, negatives):
        """
        输入:
            anchor: [batch, 2, d_model] 锚点表示（起点和终点）
            positive: [batch, 1, d_model] 正样本表示
            negatives: [batch, num_ns, d_model] 负样本表示
        输出:
            relation_vector: [batch, d_model] 关系向量
        """
        batch_size = anchor.size(0)

        anchor_mean = torch.mean(anchor, dim=1)  # [batch, d_model]

        pos_rel = positive.squeeze(1) - anchor_mean  # [batch, d_model]

        neg_rel = negatives - anchor_mean.unsqueeze(1)  # [batch, num_ns, d_model]
        neg_rel_mean = torch.mean(neg_rel, dim=1)  # [batch, d_model]

        relation_features = []
        for layer in self.relation_layers:
            combined = torch.cat([anchor_mean, pos_rel, neg_rel_mean], dim=-1)
            rel_feat = layer(combined)
            relation_features.append(rel_feat)

        relation_stack = torch.stack(relation_features, dim=1)

        attended, _ = self.relation_attention(
            relation_stack, relation_stack, relation_stack
        )

        # 平均聚合 [batch, d_model]
        relation_vector = attended.mean(dim=1)

        return self.fusion_net(relation_vector)
class GatedEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 门控机制
        self.poi_gate = nn.Linear(d_model, d_model)
        self.pos_gate = nn.Linear(d_model, d_model)
        self.time_gate = nn.Linear(d_model, d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, poi_emb, pos_emb, time_emb):
        # 计算各模态的重要性权重
        g_poi = self.sigmoid(self.poi_gate(poi_emb))
        g_pos = self.sigmoid(self.pos_gate(pos_emb))
        g_time = self.sigmoid(self.time_gate(time_emb))

        # 加权融合
        total = g_poi * poi_emb + g_pos * pos_emb + g_time * time_emb
        return total
class MLMLoss(nn.Module):
    """标准MLM损失（无焦点权重）"""

    def __init__(self, vocab_size, ignore_index=-100):
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, labels, mask):
        return self.ce_loss(
            logits.view(-1, self.vocab_size),
            labels.view(-1)
        )

class ViewContrastiveLearner(nn.Module):
    """视图对比学习模块（改进版：解决不收敛问题）"""

    def __init__(self, d_model, base_temperature=0.07, momentum=0.99):
        """
        参数:
            d_model: 模型维度
            base_temperature: 基础温度参数
            momentum: 动量编码器动量参数
        """
        super().__init__()
        self.d_model = d_model
        self.base_temperature = base_temperature
        self.momentum = momentum  # 使用更合理的动量值

        self.logit_scale = nn.Parameter(torch.tensor([1.0]) * np.log(1 / base_temperature))

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )

        # 动量编码器（使用EMA更新）
        self.momentum_projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )

        # 初始化动量编码器与主编码器相同
        self._update_momentum_encoder(0)

        # 困难负样本队列 - 减小队列大小
        self.negative_queue = None
        self.queue_size = 16384  # 更合理的队列大小
        self.queue_ptr = 0

        logger.info(f"初始化增强对比学习模块: d_model={d_model}, momentum={momentum}")

    def forward(self, view1, view2):
        """
        输入:
            view1: [batch, d_model] 第一个视图的表示
            view2: [batch, d_model] 第二个视图的表示
        输出:
            contrast_loss: 对比损失（InfoNCE损失）
        """
        batch_size = view1.size(0)

        # 1. 投影到对比空间
        q = self.projector(view1)  # 在线编码器
        k = self.momentum_projector(view2)  # 动量编码器
        # 2. 归一化
        q = F.normalize(q, p=2, dim=-1, eps=1e-8)
        k = F.normalize(k, p=2, dim=-1, eps=1e-8)
        logits = torch.mm(q, k.t()) * self.logit_scale.exp().clamp(max=100.0)
        labels = torch.arange(batch_size).to(q.device)
        loss = F.cross_entropy(logits, labels)
        self._update_momentum_encoder(self.momentum)

        self._update_negative_queue(k.detach())

        return loss

    def _update_negative_queue(self, features):
        """更新负样本队列（先进先出）"""
        if self.negative_queue is None:
            # 初始化队列
            self.negative_queue = torch.zeros(self.queue_size, self.d_model).to(features.device)

        batch_size = features.size(0)
        remaining = self.queue_size - self.queue_ptr

        if batch_size <= remaining:
            self.negative_queue[self.queue_ptr:self.queue_ptr + batch_size] = features
            self.queue_ptr = (self.queue_ptr + batch_size) % self.queue_size
        else:
            self.negative_queue[self.queue_ptr:] = features[:remaining]
            self.negative_queue[:batch_size - remaining] = features[remaining:]
            self.queue_ptr = batch_size - remaining
    def _update_momentum_encoder(self, momentum):
        """使用指数移动平均更新动量编码器"""
        with torch.no_grad():
            for param_q, param_k in zip(self.projector.parameters(), self.momentum_projector.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)

    def get_negative_samples(self, batch_size):
        """获取困难负样本"""
        if self.negative_queue is None:
            return None
        neg_indices = torch.randint(0, self.queue_size, (batch_size,))
        return self.negative_queue[neg_indices]

class RepresentationFusion(nn.Module):
    """对比学习表示和关系表示的融合模块"""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # 门控融合机制
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, cls_rep, rel_rep):
        combined = torch.cat([cls_rep, rel_rep], dim=-1)
        gate = self.gate(combined)
        fused = gate * cls_rep + (1 - gate) * rel_rep
        return self.transform(fused)

class TrajectoryGenerationLoss(nn.Module):

    def __init__(self, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, logits, targets):
        """
        输入:
            logits: [batch, seq_len, vocab_size] 预测logits
            targets: [batch, seq_len] 目标序列
        输出:
            loss: 轨迹生成损失
        """
        batch_size, seq_len, vocab_size = logits.shape
        if targets.size(1) != seq_len:
            if targets.size(1) > seq_len:
                targets = targets[:, :seq_len]
            else:
                pad_len = seq_len - targets.size(1)
                targets = F.pad(targets, (0, pad_len), value=self.pad_id)

        return self.ce_loss(
            logits.view(-1, vocab_size),
            targets.view(-1)
        )