# dirichlet_sampler.py
import numpy as np
import random
from collections import defaultdict
import pickle
import os

import logging
logger = logging.getLogger("logger")

class DirichletSampler:
    def __init__(self, dataset, params):
        self.dataset = dataset
        self.params = params
        self.classes_dict = self.build_classes_dict()

    def build_classes_dict(self):
        """构建类别到索引的字典"""
        classes = {}
        for ind, x in enumerate(self.dataset):
            _, label = x
            if label in classes:
                classes[label].append(ind)
            else:
                classes[label] = [ind]
        return classes

    def sample_dirichlet_train_data(self, num_participants, alpha, save_path=None):
        """
        迪利克雷采样并保存结果到文件 - 修正版
        """
        classes = self.classes_dict
        per_participant_list = defaultdict(list)
        num_classes = len(classes.keys())

        # 先打乱每个类别的数据
        for n in range(num_classes):
            random.shuffle(classes[n])

        total_samples = sum(len(classes[n]) for n in range(num_classes))
        logger.info(f"数据集总样本数: {total_samples}")

        allocated_per_class = [0] * num_classes  # 跟踪每个类别已分配的数量

        for n in range(num_classes):
            class_size = len(classes[n])
            remaining = class_size

            # 为当前类别生成采样比例
            proportions = np.random.dirichlet(np.array(num_participants * [alpha]))

            # 计算每个参与者应该分配的数量
            allocations = []
            remaining_float = class_size

            for user in range(num_participants):
                user_alloc_float = proportions[user] * class_size
                user_alloc = int(user_alloc_float)
                allocations.append(user_alloc)
                remaining_float -= user_alloc_float
                remaining -= user_alloc

            # 处理四舍五入导致的误差：将剩余样本分配给分配比例最高的参与者
            if remaining > 0:
                # 找到分配比例最高的参与者
                max_prop_idx = np.argmax(proportions)
                allocations[max_prop_idx] += remaining

            # 执行分配
            start_idx = 0
            for user in range(num_participants):
                num_imgs = allocations[user]
                if num_imgs > 0:
                    end_idx = start_idx + num_imgs
                    sampled_list = classes[n][start_idx:end_idx]
                    per_participant_list[user].extend(sampled_list)
                    start_idx = end_idx

            allocated_per_class[n] = sum(allocations)

        # 转换为普通字典
        indices_per_participant = dict(per_participant_list)
        dict_users = [len(indices) for indices in indices_per_participant.values()]

        # 验证
        total_allocated = sum(dict_users)
        logger.info(f"分配的总样本数: {total_allocated}")
        logger.info(f"每个类别的分配情况: {allocated_per_class}")

        if total_allocated != total_samples:
            logger.warning(f"数据分配不匹配! 总样本: {total_samples}, 已分配: {total_allocated}")

        # 保存采样结果
        if save_path:
            self.save_sampling_result(save_path, indices_per_participant, alpha)

        return indices_per_participant, dict_users

    def save_sampling_result(self, save_path, indices_dict, alpha):
        """保存采样结果到文件"""
        sampling_data = {
            'indices_per_participant': indices_dict,
            'alpha': alpha,
            'dataset_type': self.params.get('type'),
            'num_participants': len(indices_dict)
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(sampling_data, f)
        print(f"采样结果已保存到: {save_path}")

    @staticmethod
    def load_sampling_result(load_path):
        """从文件加载采样结果"""
        try:
            with open(load_path, 'rb') as f:
                sampling_data = pickle.load(f)
            return sampling_data['indices_per_participant']
        except FileNotFoundError:
            print(f"采样文件不存在: {load_path}")
            return None

