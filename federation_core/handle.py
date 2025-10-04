import copy
import os
import logging
import random
import threading
import numpy as np
import torch
import torch.utils.data
from torchvision import datasets
from torchvision.transforms import transforms
from collections import defaultdict
import pickle

from models.ResNet8 import ResNet8
from client_manager import ClientManager
from device import device

# 移除全局logger定义，改为在Handle类中为每个任务创建独立logger

class Handle:
    def __init__(self, current_time, params, task_id, name):  # 参数名改为task_id
        self.current_time = current_time
        self.model = None

        self.train_dataset = None
        self.test_dataset = None
        self.test_dataset_poisoned = None

        self.train_data = {}  # 改为字典：用户ID -> 训练数据加载器
        self.clients_data_num = {}  # 用户数据量统计
        self.test_data = None

        # 其他属性
        self.poisoned_test_data = None
        self.test_un_target_label_data = None
        self.test_target_label_data = None
        self.participants_list = None
        self.adversarial_namelist = None
        self.benign_namelist = None
        self.start_epoch = 1
        self.classes_dict = None
        self.params = params
        self.task_id = task_id  # 使用task_id
        self.name = name
        self.target = 0

        # 修改文件夹路径为 federation_core 目录下的路径
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        core_dir = os.path.join(base_dir, 'federation_core')
        self.folder_path = os.path.join(core_dir, f'saved_models/model_{self.name}_{current_time}')
        self.tinydata = []

        # 数据块相关属性
        self.data_blocks = {}  # 静态数据块映射：块号 -> 数据索引列表
        self.total_blocks = 0  # 总数据块数
        
        # 用户管理
        self.available_users_pool = []  # 可用用户池
        self.namelist = []  # 当前联邦用户列表
        self.user_data_blocks = {}  # 用户数据块映射：用户ID -> 数据块号列表

        # 创建目录
        try:
            os.makedirs(self.folder_path, exist_ok=True)  # 使用 makedirs 确保父目录存在
        except Exception as e:
            raise
        
        # 关键修改：为每个任务创建独立的logger
        self.logger = logging.getLogger(f"task_{task_id}")  # 使用任务ID创建唯一logger名称
        self.logger.propagate = False  # 禁止日志向上传播到根logger，避免重复

        # 修复日志文件路径
        log_file_path = os.path.join(self.folder_path, 'log.txt')
        
        # 检查并添加文件处理器（仅当前任务的logger）
        has_file_handler = any(
            isinstance(handler, logging.FileHandler) and handler.baseFilename == log_file_path
            for handler in self.logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(filename=log_file_path)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # 控制台输出处理器（每个任务可选，或全局统一）
        has_stream_handler = any(
            isinstance(handler, logging.StreamHandler)
            for handler in self.logger.handlers
        )
        if not has_stream_handler:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

        self.logger.setLevel(logging.DEBUG)
        self.logger.info(f'current path: {self.folder_path}')
        
        if not self.params.get('environment_name', False):
            self.params['environment_name'] = self.name
        self.params['current_time'] = self.current_time
        self.params['folder_path'] = self.folder_path

        # 修复采样文件路径
        self.sampling_file_path = os.path.join(core_dir, 'sampling_results/sampling_cifar10.pkl')

        # 修复客户端管理器路径
        user_db_path = os.path.join(core_dir, 'user_database.json')
        self.client_manager = ClientManager(user_db_path)
        self._user_lock = threading.RLock()
        
        self.logger.info("联邦学习环境初始化完成 - 初始用户数: 0")

    def load_data(self):
        """加载数据和数据块划分"""
        self.logger.info('Loading data')
        
        # 修改数据路径为 federation_core 目录下的路径
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        core_dir = os.path.join(base_dir, 'federation_core')
        dataPath = os.path.join(core_dir, 'data')
        
        # 确保数据目录存在
        os.makedirs(dataPath, exist_ok=True)
        
        # 数据加载
        if self.params['type'] == 'CIFAR10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.train_dataset = datasets.CIFAR10(dataPath, train=True, download=True, transform=transform_train)
            self.test_dataset = datasets.CIFAR10(dataPath, train=False, download=True, transform=transform_test)
            self.test_dataset_poisoned = datasets.CIFAR10(dataPath, train=False, download=True, transform=transform_test)
        elif self.params['type'] == 'MNIST':
            transform_train = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.train_dataset = datasets.MNIST(dataPath, train=True, download=True, transform=transform_train)
            self.test_dataset = datasets.MNIST(dataPath, train=False, download=True, transform=transform_test)
            self.test_dataset_poisoned = datasets.MNIST(dataPath, train=False, download=True, transform=transform_test)
        else:
            self.logger.info('no this type!!!!!')
            return

        self.logger.info('reading data done')
        self.classes_dict = self.build_classes_dict()
        
        # 加载数据块划分
        self._load_data_blocks()
        
        # 加载用户数据块分配信息
        self._load_user_data_assignments()

        # 构建测试数据
        self.test_data = self.get_test()
        
        self.logger.info('数据加载完成')

    def _load_data_blocks(self):
        """从文件加载数据块划分"""
        try:
            # 确保文件存在
            if not os.path.exists(self.sampling_file_path):
                self.logger.error(f'数据块文件不存在: {self.sampling_file_path}')
                # 创建空的采样文件结构作为临时解决方案
                empty_data = {'indices_per_participant': {}}
                os.makedirs(os.path.dirname(self.sampling_file_path), exist_ok=True)
                with open(self.sampling_file_path, 'wb') as f:
                    pickle.dump(empty_data, f)
                self.logger.info(f'创建了空的采样文件: {self.sampling_file_path}')
            
            with open(self.sampling_file_path, 'rb') as f:
                data_blocks_data = pickle.load(f)
                self.data_blocks = data_blocks_data['indices_per_participant']
                self.total_blocks = len(self.data_blocks)
            self.logger.info(f"加载数据块完成，共 {self.total_blocks} 个数据块")
        except Exception as e:
            self.logger.error(f'加载数据块失败: {e}')
            # 创建空的数据块作为后备
            self.data_blocks = {}
            self.total_blocks = 0

    def _load_user_data_assignments(self):
        """从数据库加载用户数据块分配信息"""
        with self._user_lock:
            try:
                all_users = self.client_manager.get_all_users()
                self.logger.info(f"数据库中找到 {len(all_users)} 个用户")
                
                for uid in all_users:
                    user_id = int(uid)
                    # 从数据库读取用户的数据块分配
                    block_ids = self.client_manager.get_user_data_indices(uid)
                    self.user_data_blocks[user_id] = block_ids
                    # 将用户添加到可用池
                    self.available_users_pool.append(user_id)
                    self.logger.debug(f"用户 {user_id} 分配了数据块: {block_ids}")
                
                self.logger.info(f"加载用户数据分配完成，共 {len(self.available_users_pool)} 个可用用户")
            except Exception as e:
                self.logger.error(f"加载用户数据分配失败: {e}")
                # 创建一些测试用户作为后备
                self._create_test_users()

    def _create_test_users(self):
        """创建测试用户数据（用于开发和测试）"""
        self.logger.info("创建测试用户数据...")
        test_users = [
            (1, "测试用户1", [0, 1, 2]),
            (2, "测试用户2", [3, 4, 5]),
            (3, "测试用户3", [6, 7, 8]),
            (4, "测试用户4", [9, 10, 11]),
            (5, "测试用户5", [12, 13, 14]),
        ]
        
        for user_id, user_name, block_ids in test_users:
            self.user_data_blocks[user_id] = block_ids
            self.available_users_pool.append(user_id)
            # 添加到客户端管理器
            self.client_manager.add_user(user_id, user_name, block_ids)
        
        self.logger.info(f"创建了 {len(test_users)} 个测试用户")

    def _build_train_data(self):
        """为当前联邦用户构建训练数据 - 构建为字典"""
        with self._user_lock:
            train_loaders = {}  # 改为字典，键为用户ID，值为训练数据加载器
            clients_data_num = {}
            
            self.logger.info(f"开始为 {len(self.namelist)} 个联邦用户构建训练数据")
            
            for user_id in self.namelist:
                # 获取用户的数据块号
                user_block_ids = self.user_data_blocks.get(user_id, [])
                self.logger.info(f"用户 {user_id} 的数据块: {user_block_ids}")
                
                # 将数据块号转换为具体的数据索引
                all_indices = []
                for block_id in user_block_ids:
                    if block_id in self.data_blocks:
                        all_indices.extend(self.data_blocks[block_id])
                    else:
                        self.logger.warning(f"用户 {user_id} 的数据块 {block_id} 不存在")
                
                if all_indices:
                    try:
                        train_loader = self.get_train(all_indices)
                        train_loaders[user_id] = train_loader  # 使用用户ID作为键
                        clients_data_num[user_id] = len(all_indices)
                        self.logger.info(f"用户 {user_id} 有 {len(all_indices)} 个训练样本")
                    except Exception as e:
                        self.logger.error(f"为用户 {user_id} 构建训练数据时发生错误: {e}")
                        continue
                else:
                    self.logger.warning(f"用户 {user_id} 没有有效数据")
                    # 如果没有数据，使用随机数据作为后备
                    try:
                        all_indices = list(range(100))  # 使用前100个样本作为测试
                        train_loader = self.get_train(all_indices)
                        train_loaders[user_id] = train_loader
                        clients_data_num[user_id] = len(all_indices)
                        self.logger.info(f"用户 {user_id} 使用测试数据，样本数: {len(all_indices)}")
                    except Exception as e:
                        self.logger.error(f"为用户 {user_id} 创建测试数据失败: {e}")
            
            self.train_data = train_loaders  # 现在是一个字典
            self.clients_data_num = clients_data_num
            self.logger.info(f"训练数据构建完成，共 {len(self.train_data)} 个联邦用户")

    def build_classes_dict(self):
        classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if label in classes:
                classes[label].append(ind)
            else:
                classes[label] = [ind]
        return classes

    def get_train(self, indices):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.params['batch_size'],
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            pin_memory=True, 
            num_workers=0
        )
        return train_loader

    def get_test(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.params['batch_size'],
            shuffle=False
        )
        return test_loader

    def create_model(self):
        model = None
        if self.params['model'] == 'CNN':
            if self.params['type'] == 'MNIST':
                model = CNNMnist()
            elif self.params['type'] == 'CIFAR10':
                model = CNNCifar10()
        elif self.params['model'] == 'r8':
            model = ResNet8()
        elif self.params['model'] == 'r18':
            model = ResNet18()
        elif self.params['model'] == 'r34':
            model = ResNet34()
        model = model.to(device)
        self.model = model

    # 联邦用户管理方法
    def add_user_to_federation(self, user_id):
        """向联邦中添加用户"""
        with self._user_lock:
            self.logger.info(f"尝试添加用户 {user_id} 到联邦")
            
            if user_id in self.namelist:
                self.logger.warning(f"用户 {user_id} 已在联邦中")
                return False
            
            # 检查用户是否有数据分配
            if user_id not in self.user_data_blocks:
                self.logger.warning(f"用户 {user_id} 没有数据分配")
                return False
            
            # 检查用户是否在可用池中
            if user_id not in self.available_users_pool:
                self.logger.warning(f"用户 {user_id} 不在可用池中")
                return False
            
            # 添加到联邦
            self.namelist.append(user_id)
            
            # 从可用池移除
            self.available_users_pool.remove(user_id)
            
            # 重新构建训练数据
            self._build_train_data()
            
            self.logger.info(f"用户 {user_id} 已成功加入联邦")
            return True

    def remove_user_from_federation(self, user_id):
        """从联邦中移除用户"""
        with self._user_lock:
            self.logger.info(f"尝试从联邦中移除用户 {user_id}")
            
            if user_id not in self.namelist:
                self.logger.warning(f"用户 {user_id} 不在联邦中")
                return False
            
            # 从联邦移除
            self.namelist.remove(user_id)
            
            # 放回可用池
            if user_id not in self.available_users_pool:
                self.available_users_pool.append(user_id)
            
            # 重新构建训练数据
            self._build_train_data()
            
            self.logger.info(f"用户 {user_id} 已成功退出联邦")
            return True

    def add_random_user(self):
        """随机添加用户到联邦"""
        with self._user_lock:
            if not self.available_users_pool:
                self.logger.warning("没有可用的用户可以添加")
                return False
            
            user_id = random.choice(self.available_users_pool)
            self.logger.info(f"随机选择用户 {user_id} 添加到联邦")
            return self.add_user_to_federation(user_id)

    def remove_random_user(self):
        """从联邦中随机移除用户"""
        with self._user_lock:
            if len(self.namelist) <= 2:
                self.logger.warning("用户数过少，不能移除")
                return False
            
            user_to_remove = random.choice(self.namelist)
            self.logger.info(f"随机选择用户 {user_to_remove} 从联邦中移除")
            return self.remove_user_from_federation(user_to_remove)

    def get_status(self):
        """获取系统状态"""
        with self._user_lock:
            return {
                "current_epoch": self.start_epoch - 1,
                "active_users": len(self.namelist),
                "available_users": len(self.available_users_pool),
                "total_registered_users": len(self.user_data_blocks)
            }