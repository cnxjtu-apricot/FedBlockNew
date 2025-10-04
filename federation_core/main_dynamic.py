import threading
import time
import random
import logging
import yaml
from handle import Handle
from algorithm import FedAvg

logger = logging.getLogger("logger")

class DynamicFederation:
    def __init__(self, config_path='cifar_params.yaml'):
        # 加载配置
        with open(config_path, 'r') as f:
            self.params = yaml.safe_load(f)
        
        # 初始化联邦学习环境
        current_time = str(int(time.time()))
        self.handle = Handle(current_time, self.params, "dynamic_federation")
        
        # 加载数据和模型
        self.handle.load_data()
        self.handle.create_model()
        
        # 线程控制
        self.training_thread = None
        self.user_management_thread = None
        self.is_training = False
        self.is_user_management = False
        self.training_paused = False
        
        logger.info("动态联邦学习环境初始化完成")

    def start_federation(self):
        """启动联邦学习系统"""
        logger.info("启动动态联邦学习系统...")
        
        # 启动训练线程
        self.is_training = True
        self.training_thread = threading.Thread(target=self._training_loop)
        self.training_thread.daemon = True
        self.training_thread.start()
        
        # 启动用户管理线程
        self.is_user_management = True
        self.user_management_thread = threading.Thread(target=self._user_management_loop)
        self.user_management_thread.daemon = True
        self.user_management_thread.start()
        
        logger.info("联邦训练和用户管理线程已启动")

    def stop_federation(self):
        """停止联邦学习系统"""
        logger.info("正在停止联邦学习系统...")
        
        self.is_training = False
        self.is_user_management = False
        
        if self.training_thread:
            self.training_thread.join(timeout=10)
        if self.user_management_thread:
            self.user_management_thread.join(timeout=10)
            
        logger.info("联邦学习系统已停止")

    def _training_loop(self):
        """训练循环"""
        while self.is_training:
            current_users = len(self.handle.namelist)
                
            # 检查用户数量条件
            if current_users < 2:
                if not self.training_paused:
                    logger.warning(f"当前用户数 {current_users} 少于2人，训练挂起")
                    self.training_paused = True
                time.sleep(5)
                continue
            
            if self.training_paused:
                logger.info(f"当前用户数 {current_users} 达到要求，恢复训练")
                self.training_paused = False
        
            # 执行联邦训练
            try:
                logger.info(f"开始第 {self.handle.start_epoch} 轮联邦训练，当前用户数: {current_users}")
                FedAvg(self.handle)
                self.handle.start_epoch += 1
                
                # 检查是否达到最大训练轮数
                if self.handle.start_epoch > self.params['epochs']:
                    logger.info("达到最大训练轮数，停止训练")
                    self.is_training = False
                    break
                    
            except Exception as e:
                logger.error(f"训练过程中发生错误: {e}")
                time.sleep(10)

    def _user_management_loop(self):
        """用户管理循环"""
        check_interval = 10
        
        # 初始添加几个用户
        self._add_initial_users()
        
        while self.is_user_management:
            try:
                status = self.handle.get_status()
                current_users = status["active_users"]
                    
                # 决定是否添加用户
                if current_users < 9 and random.random() < 0.4:
                    self.handle.add_random_user()
                
                # 决定是否移除用户
                if current_users > 3 and random.random() < 0.3:
                    self.handle.remove_random_user()
            
                # 记录当前状态
                logger.info(f"用户管理检查完成 - 活跃用户: {current_users}, 可用用户: {status['available_users']}")
                
            except Exception as e:
                logger.error(f"用户管理过程中发生错误: {e}")
            
            time.sleep(check_interval)

    def _add_initial_users(self):
        """初始添加几个用户"""
        initial_users_count = 6
        for _ in range(initial_users_count):
            if not self.handle.add_random_user():
                break
        logger.info(f"初始添加了 {initial_users_count} 个用户")

    def get_status(self):
        """获取系统状态"""
        return self.handle.get_status()

def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    federation = DynamicFederation()
    
    try:
        federation.start_federation()
        
        while federation.is_training:
            status = federation.get_status()
            logger.info(f"系统状态: 轮次{status['current_epoch']}, "
                       f"活跃用户{status['active_users']}, "
                       f"可用用户池{status['available_users']}, "
                       f"训练{'暂停' if federation.training_paused else '进行中'}")
            time.sleep(30)
            
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止系统...")
    finally:
        federation.stop_federation()
        logger.info("系统已完全停止")

if __name__ == "__main__":
    main()