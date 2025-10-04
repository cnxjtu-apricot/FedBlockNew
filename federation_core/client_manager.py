import json
import os

class ClientManager:
    def __init__(self, db_path='user_database.json'):
        # 确保传入的是绝对路径，或者转换为绝对路径
        if not os.path.isabs(db_path):
            # 如果是相对路径，基于当前工作目录转换为绝对路径
            db_path = os.path.abspath(db_path)
        
        # 确保目录存在
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            
        self.db_path = db_path
        self.user_info = self._load_user_database()
    
    def _load_user_database(self):
        """从JSON文件加载用户数据库"""
        try:
            with open(self.db_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"user_info": {}}
    
    def _save_user_database(self):
        """保存用户数据库到JSON文件"""
        with open(self.db_path, 'w') as f:
            json.dump(self.user_info, f, indent=2)
    
    def get_user_data_indices(self, user_id):
        """获取用户的数据块索引"""
        if str(user_id) in self.user_info["user_info"]:
            return self.user_info["user_info"][str(user_id)]["assigned_data_indices"]
        return None
    
    def get_all_users(self):
        """获取所有用户ID"""
        return list(self.user_info["user_info"].keys())
    
    def add_user(self, user_id, user_name, data_indices, initial_coins=100):
        """添加新用户"""
        user_id_str = str(user_id)
        if user_id_str not in self.user_info["user_info"]:
            self.user_info["user_info"][user_id_str] = {
                "user_id": user_id,
                "user_name": user_name,
                "virtual_coins": initial_coins,
                "assigned_data_indices": data_indices,
                "data_block_count": len(data_indices)
            }
            self._save_user_database()
            return True
        return False
    
    def remove_user(self, user_id):
        """移除用户"""
        user_id_str = str(user_id)
        if user_id_str in self.user_info["user_info"]:
            del self.user_info["user_info"][user_id_str]
            self._save_user_database()
            return True
        return False