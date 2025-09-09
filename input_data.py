import math
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
from collections import defaultdict
import pickle
import time
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class TrajectoryDataProcessor:
    def __init__(self, city, max_seq_len=20, base_dir=None):
        self.city = city
        self.max_seq_len = min(max_seq_len, 20)
        if base_dir is None:
            self.base_dir = Path.cwd()
        else:
            self.base_dir = Path(base_dir)
        self.poi_file = self.base_dir / 'origin_data' / f'poi-{city}.csv'
        self.traj_file = self.base_dir / 'processed_data' / f'{city}_process.csv'
        self.vocab_file = self.base_dir / 'vocab_files' / f'{city}-vocab.txt'
        self.distance_matrix_file = self.base_dir / 'distance_matrix' / f'{city}_distance_matrix.pkl'
        os.makedirs(self.poi_file.parent, exist_ok=True)
        os.makedirs(self.vocab_file.parent, exist_ok=True)
        os.makedirs(self.distance_matrix_file.parent, exist_ok=True)
        self.special_tokens = ['[PAD]', '[UNK]', '[MASK]', '[CLS]', '[SEP]', '[SOS]', '[EOS]']
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.mask_token = '[MASK]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.sos_token = '[SOS]'
        self.eos_token = '[EOS]'
        self.poi_df = None
        self.traj_df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.token_to_id = {}
        self.id_to_token = {}
        self.pad_id = None
        self.unk_id = None
        self.mask_id = None
        self.cls_id = None
        self.sep_id = None
        self.sos_id = None
        self.eos_id = None
        self.distance_matrix = None
        self.poi_coords = {}
        self.user_history = defaultdict(list)
        self.transition_matrix = None
        self.load_data()
        self.split_raw_data()
        self.create_vocab()
        self.compute_distance_matrix()
        self.process_trajectories()
        self.generate_transition_matrix()
        self.load_user_history()
        self.compute_trajectory_length_map()
        self.augment_training_set()
        self.random_seed = self.generate_random_seed()
    def _calculate_distance(self, lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371
        return c * r * 1000

    def compute_distance_matrix(self):
        if self.distance_matrix_file.exists():
            with open(self.distance_matrix_file, 'rb') as f:
                self.distance_matrix = pickle.load(f)
            return
        if self.poi_df is None:
            self.load_data()
        pois = self.poi_df[['poiID', 'poiLon', 'poiLat']].copy()
        pois['poiID'] = pois['poiID'].astype(str)
        self.poi_coords = {row['poiID']: (row['poiLon'], row['poiLat']) for _, row in pois.iterrows()}
        poi_ids = list(self.poi_coords.keys())
        n = len(poi_ids)
        distance_matrix = np.zeros((n, n))
        for i in tqdm(range(n), desc="计算距离矩阵"):
            for j in range(i + 1, n):
                id_i = poi_ids[i]
                id_j = poi_ids[j]
                lon1, lat1 = self.poi_coords[id_i]
                lon2, lat2 = self.poi_coords[id_j]
                dist = self._calculate_distance(lon1, lat1, lon2, lat2)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        self.distance_matrix = pd.DataFrame(distance_matrix, index=poi_ids, columns=poi_ids)
        with open(self.distance_matrix_file, 'wb') as f:
            pickle.dump(self.distance_matrix, f)
    def load_data(self):
        self.poi_df = pd.read_csv(self.poi_file, encoding="gbk")
        self.traj_df = pd.read_csv(self.traj_file)
        required_cols = ['userID', 'poi_sequence', 'start_time', 'end_time']
        for col in required_cols:
            if col not in self.traj_df.columns:
                raise ValueError(f"轨迹数据缺少必要列: {col}")
    def split_raw_data(self):
        indices = list(range(len(self.traj_df)))
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=0.3
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=1 / 3
        )
        self.train_df = self.traj_df.iloc[train_idx].copy()
        self.val_df = self.traj_df.iloc[val_idx].copy()
        self.test_df = self.traj_df.iloc[test_idx].copy()
        return self.train_df, self.val_df, self.test_df
    def generate_random_seed(self):
        """生成基于时间的随机种子"""
        return int(time.time() * 1000) % 1000000

    def create_vocab(self):
        poi_ids = sorted(self.poi_df['poiID'].unique().astype(str).tolist())
        missing_pois = set(self.poi_df['poiID'].astype(str)) - set(poi_ids)
        if missing_pois:
            logger.warning(f"词汇表缺失{len(missing_pois)}个POI")
        vocab = self.special_tokens + poi_ids
        with open(self.vocab_file, 'w') as f:
            f.write('\n'.join(vocab))
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.pad_id = self.token_to_id[self.pad_token]
        self.unk_id = self.token_to_id[self.unk_token]
        self.mask_id = self.token_to_id[self.mask_token]
        self.cls_id = self.token_to_id[self.cls_token]
        self.sep_id = self.token_to_id[self.sep_token]
        self.sos_id = self.token_to_id[self.sos_token]
        self.eos_id = self.token_to_id[self.eos_token]
        self.poi_num = len(poi_ids)  
    def process_trajectories(self):
        processed_counts = {'train': 0, 'val': 0, 'test': 0}

        for df_name in ['train_df', 'val_df', 'test_df']:
            df = getattr(self, df_name)
            original_count = len(df)
            df['poi_list'] = df['poi_sequence'].apply(
                lambda x: [str(p) for p in x.strip().split()]
            )
            df = df[df['poi_list'].apply(len) >= 3] 
            df['poi_list'] = df['poi_list'].apply(
                lambda traj: traj[:self.max_seq_len] if len(traj) > self.max_seq_len else traj
            )
            df['start_timestamp'] = df['start_time']
            df['end_timestamp'] = df['end_time']
            df['start_time'] = pd.to_datetime(df['start_time'], unit='s')
            df['end_time'] = pd.to_datetime(df['end_time'], unit='s')
            df['start_hour'] = df['start_time'].dt.hour
            df['end_hour'] = df['end_time'].dt.hour
            def onehot_hour(hour):
                vec = np.zeros(24)
                vec[int(hour)] = 1
                return vec.tolist()
            df['start_hour_onehot'] = df['start_hour'].apply(onehot_hour)
            df['end_hour_onehot'] = df['end_hour'].apply(onehot_hour)
            df['start_poi'] = df['poi_list'].apply(lambda x: x[0])
            df['end_poi'] = df['poi_list'].apply(lambda x: x[-1])
            setattr(self, df_name, df)
            processed_counts[df_name.split('_')[0]] = len(df)
    def tokenize_traj(self, poi_list):
        return [
            self.token_to_id.get(poi, self.unk_id)
            for poi in poi_list
        ]
    def pad_and_mask(self, token_ids, max_len=None):
        max_len = max_len or self.max_seq_len
        length = len(token_ids)
        if length > max_len:
            token_ids = token_ids[:max_len]
            mask = [1] * max_len
            valid_len = max_len
        else:
            pad_len = max_len - length
            token_ids = token_ids + [self.pad_id] * pad_len
            mask = [1] * length + [0] * pad_len
            valid_len = length
        return token_ids, mask, valid_len
    def prepare_pretrain_data(self):
        pretrain_data = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
            'time_features': [],
            'view_pair_id': [],
            'position_ids': [],
            'relative_positions': [],
            'valid_lengths': []
        }
        df = self.train_df.copy()
        view_pair_counter = 0
        for _, row in df.iterrows():
            poi_list = row['poi_list']
            if len(poi_list) < 3:
                continue
            time_features = np.array(row['start_hour_onehot'] + row['end_hour_onehot'])
            token_ids = self.tokenize_traj(poi_list)
            seq_len = len(token_ids)
            position_ids = list(range(seq_len))
            rel_positions = []
            for i in range(seq_len):
                rel_row = [min(10, abs(i - j)) for j in range(seq_len)]
                rel_positions.append(rel_row)
            for i in range(2):
                view = self.augment_trajectory(token_ids.copy())
                masked_ids, labels = self.apply_masking(view)
                input_ids = [self.cls_id] + masked_ids + [self.sep_id]
                label_ids = [-100] + labels + [-100]
                padded_ids, attn_mask, _ = self.pad_and_mask(input_ids, self.max_seq_len)
                padded_labels, _, _ = self.pad_and_mask(label_ids, self.max_seq_len)
                padded_position_ids = position_ids[:self.max_seq_len]  
                if len(padded_position_ids) < self.max_seq_len:
                    padded_position_ids += [0] * (self.max_seq_len - len(padded_position_ids))
                padded_rel_positions = []
                for i in range(self.max_seq_len):
                    if i < len(rel_positions):
                        row = rel_positions[i][:self.max_seq_len]
                        if len(row) < self.max_seq_len:
                            row += [0] * (self.max_seq_len - len(row))
                    else:
                        row = [0] * self.max_seq_len
                    padded_rel_positions.append(row)
                padded_rel_positions = padded_rel_positions[:self.max_seq_len]
                if len(padded_rel_positions) < self.max_seq_len:
                    padded_rel_positions += [[0] * self.max_seq_len] * (self.max_seq_len - len(padded_rel_positions))
                pretrain_data['input_ids'].append(padded_ids)
                pretrain_data['attention_mask'].append(attn_mask)
                pretrain_data['labels'].append(padded_labels)
                pretrain_data['time_features'].append(time_features)
                pretrain_data['view_pair_id'].append(view_pair_counter)
                pretrain_data['position_ids'].append(padded_position_ids)
                pretrain_data['relative_positions'].append(padded_rel_positions)
                pretrain_data['valid_lengths'].append(min(seq_len, self.max_seq_len))
            view_pair_counter += 1
        pretrain_data['input_ids'] = np.array(pretrain_data['input_ids'])
        pretrain_data['attention_mask'] = np.array(pretrain_data['attention_mask'])
        pretrain_data['labels'] = np.array(pretrain_data['labels'])
        pretrain_data['time_features'] = np.array(pretrain_data['time_features'])
        pretrain_data['view_pair_id'] = np.array(pretrain_data['view_pair_id'])
        pretrain_data['position_ids'] = np.array(pretrain_data['position_ids'])
        pretrain_data['relative_positions'] = np.array(pretrain_data['relative_positions'])
        pretrain_data['valid_lengths'] = np.array(pretrain_data['valid_lengths'])
        return pretrain_data
    def augment_trajectory(self, token_ids):
        """轨迹增强方法（保持顺序不变）"""
        original_length = len(token_ids)
        if original_length < 3:
            return token_ids
        start_token = token_ids[0]
        end_token = token_ids[-1]
        mid_tokens = token_ids[1:-1]
        mid_len = len(mid_tokens)
        if mid_len == 0:
            return token_ids
        method_weights = {
            'no_aug': 0.2,  
            'token_cutoff': 0.15,  
            # 'dropout': 0.15, 
            'insert': 0.20, 
            'mask': 0.2,  
            'replace': 0.25 
        }
        method = random.choices(
            list(method_weights.keys()),
            weights=list(method_weights.values())
        )[0]
        if method == 'no_aug':
            return token_ids
        elif method == 'token_cutoff':
            keep_len = random.randint(max(1, mid_len // 2), mid_len)  
            start_idx = random.randint(0, mid_len - keep_len)
            selected_mid = mid_tokens[start_idx:start_idx + keep_len]
            return [start_token] + selected_mid + [end_token]
        elif method == 'dropout':
            keep_prob = random.uniform(0.6, 0.8)
            new_mid = [tok for tok in mid_tokens if random.random() < keep_prob]
            if not new_mid:
                new_mid = [random.choice(mid_tokens)]
            return [start_token] + new_mid + [end_token]
        elif method == 'insert' and original_length < self.max_seq_len - 1:
            insert_idx = random.randint(0, mid_len)
            insert_token = random.choice(mid_tokens)
            new_mid = mid_tokens.copy()
            new_mid.insert(insert_idx, insert_token)
            return [start_token] + new_mid[:self.max_seq_len - 2] + [end_token]
        elif method == 'mask' and mid_len > 1:
            num_mask = random.randint(1, min(3, mid_len))
            mask_indices = set(random.sample(range(mid_len), num_mask))
            new_mid = [
                self.mask_id if i in mask_indices else tok
                for i, tok in enumerate(mid_tokens)
            ]
            return [start_token] + new_mid + [end_token]
        elif method == 'replace' and mid_len > 1:
            replace_idx = random.randint(0, mid_len - 1)
            candidates = [tok for tok in mid_tokens if tok != mid_tokens[replace_idx]]
            if candidates:
                new_token = random.choice(candidates)
                new_mid = mid_tokens.copy()
                new_mid[replace_idx] = new_token
                return [start_token] + new_mid + [end_token]
        return token_ids

    def apply_masking(self, token_ids):
        if len(token_ids) < 3:
            return token_ids, [-100] * len(token_ids)
        mid_indices = list(range(1, len(token_ids) - 1))
        if not mid_indices:
            return token_ids, [-100] * len(token_ids)
        num_mask = max(1, min(len(mid_indices), int(len(mid_indices) * 0.3)))  # 30%掩码率
        mask_indices = random.sample(mid_indices, num_mask)
        masked_ids = token_ids.copy()
        labels = [-100] * len(token_ids)
        for idx in mask_indices:
            if random.random() < 0.8:
                masked_ids[idx] = self.mask_id
            elif random.random() < 0.5:
                masked_ids[idx] = random.choice(list(self.token_to_id.values()))
            labels[idx] = token_ids[idx]
        return masked_ids, labels
    def generate_transition_matrix(self, smoothing=0.01, print_matrix=True):

        full_df = pd.concat([self.train_df, self.val_df, self.test_df])

        all_pois = full_df['poi_list'].explode().unique().astype(str)
        poi_count = len(all_pois)
        transition_dict = defaultdict(lambda: defaultdict(float))
        transition_count = 0
        for _, row in tqdm(full_df.iterrows(), total=len(full_df), desc="统计转移频率（完整数据）"):
            traj = row['poi_list']
            for i in range(len(traj) - 1):
                from_poi = str(traj[i])
                to_poi = str(traj[i + 1])
                transition_dict[from_poi][to_poi] += 1
                transition_count += 1

        transition_df = pd.DataFrame(
            index=all_pois,
            columns=all_pois,
            data=0.0
        )

        for from_poi, targets in transition_dict.items():
            for to_poi, count in targets.items():
                if from_poi in transition_df.index and to_poi in transition_df.columns:
                    transition_df.at[from_poi, to_poi] = count
        n = len(all_pois)
        for poi in tqdm(transition_df.index, desc="平滑和归一化"):
            row_sum = transition_df.loc[poi].sum()
            transition_df.loc[poi] = (transition_df.loc[poi] + smoothing) / (row_sum + smoothing * n)
        self.transition_matrix = transition_df
        self.transition_stats = {
            "total_transitions": transition_count,
            "unique_from_pois": len(transition_df.index),
            "unique_to_pois": len(transition_df.columns),
            "sparsity": 1 - (transition_count / (poi_count * poi_count))
        }
        return transition_df
    def load_user_history(self):
        if 'userID' not in self.train_df.columns:
            logger.warning("轨迹数据缺少userID列，无法加载用户历史轨迹")
            return {}
        combined_df = pd.concat([self.train_df, self.val_df, self.test_df])
        combined_df = combined_df.sort_values('start_timestamp')
        grouped = combined_df.groupby('userID')
        for user_id, group in tqdm(grouped, desc="处理用户轨迹"):
            user_trajs = group['poi_list'].tolist()
            self.user_history[user_id] = user_trajs
        return self.user_history
    def compute_trajectory_length_map(self):
        combined_df = pd.concat([self.train_df, self.val_df, self.test_df])
        length_counts = defaultdict(list)
        for _, row in combined_df.iterrows():
            poi_list = row['poi_list']
            if len(poi_list) < 2:
                continue
            start_id = poi_list[0]
            end_id = poi_list[-1]
            length = len(poi_list)
            length_counts[(start_id, end_id)].append(length)
        self.traj_length_map = {}
        for key, lengths in length_counts.items():
            avg_length = sum(lengths) / len(lengths)
            self.traj_length_map[key] = math.floor(avg_length)  
        all_lengths = [len(traj) for traj in combined_df['poi_list']]
        if all_lengths:
            self.global_avg_length = math.floor(sum(all_lengths) / len(all_lengths))
        else:
            self.global_avg_length = 3  
            logger.warning("未找到有效轨迹，使用默认全局平均长度=3")
        return self.traj_length_map
    def prepare_finetune_data(self):
        """准备微调数据（轨迹生成）"""
        finetune_data = {'train': {}, 'val': {}, 'test': {}}

        for dataset in ['train', 'val', 'test']:
            df = getattr(self, f'{dataset}_df')
            if 'is_augmented' in df.columns:
                df = df[~df['is_augmented']]
            data = {
                'condition': [],
                'input_ids': [],
                'attention_mask': [],
                'target_ids': [],
                'time_features': []
            }

            for _, row in df.iterrows():
                poi_list = row['poi_list']
                start_poi = row['start_poi']
                end_poi = row['end_poi']

                start_id = self.token_to_id.get(start_poi, self.unk_id)
                end_id = self.token_to_id.get(end_poi, self.unk_id)

                poi_list_ids = [self.token_to_id.get(poi, self.unk_id) for poi in poi_list]

                input_seq = [self.sos_id] + poi_list_ids
                target_seq = poi_list_ids + [self.eos_id]

                max_len = max(len(input_seq), len(target_seq))
                if len(input_seq) < max_len:
                    input_seq += [self.pad_id] * (max_len - len(input_seq))
                else:
                    input_seq = input_seq[:max_len]
                if len(target_seq) < max_len:
                    target_seq += [self.pad_id] * (max_len - len(target_seq))
                else:
                    target_seq = target_seq[:max_len]

                attention_mask = [1] * len(poi_list_ids) + [0] * (max_len - len(poi_list_ids))

                if max_len > self.max_seq_len:
                    input_seq = input_seq[:self.max_seq_len]
                    target_seq = target_seq[:self.max_seq_len]
                    attention_mask = attention_mask[:self.max_seq_len]
                else:
                    pad_len = self.max_seq_len - max_len
                    input_seq += [self.pad_id] * pad_len
                    target_seq += [self.pad_id] * pad_len
                    attention_mask += [0] * pad_len

                time_features = np.array(row['start_hour_onehot'] + row['end_hour_onehot'])

                data['condition'].append((start_id, end_id))
                data['input_ids'].append(input_seq)
                data['attention_mask'].append(attention_mask)
                data['target_ids'].append(target_seq)
                data['time_features'].append(time_features)
            finetune_data[dataset] = data
        return finetune_data


    def augment_training_set(self):
        original_train_df = self.train_df.copy()
        augmented_rows = []
        for _, row in tqdm(original_train_df.iterrows(), total=len(original_train_df), desc="生成增强轨迹"):
            poi_list = row['poi_list']
            if len(poi_list) < 3:
                continue
            token_ids = self.tokenize_traj(poi_list)
            for _ in range(3):
                augmented_ids = self.augment_trajectory(token_ids.copy())
                augmented_pois = [self.id_to_token.get(id_, self.unk_token) for id_ in augmented_ids]
                new_row = row.copy()
                new_row['poi_sequence'] = ' '.join(augmented_pois)
                new_row['poi_list'] = augmented_pois
                new_row['is_augmented'] = True

                augmented_rows.append(new_row)

        if augmented_rows:
            augmented_df = pd.DataFrame(augmented_rows)
            self.train_df = pd.concat([original_train_df, augmented_df], ignore_index=True)
        else:
            logger.warning("未生成任何增强轨迹")

        return self.train_df