import copy
import logging
import random
import time
import torch
import train
import test
from models.ResNet8 import ResNet8
from device import device
# 移除全局logger定义，使用handle传入的logger

def get_clients(epoch, handle):
    agent_name_keys = handle.namelist
    handle.logger.info(f'Server Epoch:{epoch} choose agents : {agent_name_keys}.')  # 使用handle.logger
    return agent_name_keys


def Aggregation(w_ori, w_list, lens, beta, eta, defence_method, params):
    """优化聚合函数，使用向量化操作减少循环"""
    # 预计算所有键，避免在循环中重复获取
    keys = [k for k in w_ori.keys() if w_ori[k].dtype != torch.int64]
    
    # 使用向量化操作替代循环
    with torch.no_grad():
        # 计算加权差异
        weighted_diffs = []
        for i in range(len(w_list)):
            diff_dict = {}
            for k in keys:
                diff_dict[k] = (w_list[i][k] - w_ori[k]) * beta[i]
            weighted_diffs.append(diff_dict)
        
        # 计算平均值
        w_avg = {}
        for k in keys:
            stacked = torch.stack([wd[k] for wd in weighted_diffs])
            w_avg[k] = torch.mean(stacked, dim=0)
        
        # 更新原始权重
        for k in keys:
            w_ori[k] += eta * w_avg[k]
    
    return w_ori


# tasks.py 修改 FedAvg 函数

def FedAvg(handle):
    # 预加载模型到设备，避免重复操作
    handle.model.to(device)
    
    for epoch in range(handle.start_epoch, handle.params['epochs'] + 1):
        if handle.params['lr_decay'] is True:
            if epoch % handle.params['lr_decay_epoch'] == 0:
                handle.params['lr'] = handle.params['lr'] * handle.params['lr_decay_gamma']
                handle.params['poison_lr'] = handle.params['poison_lr'] * handle.params['lr_decay_gamma']
        
        start_time = time.time()
        agent_name_keys = get_clients(epoch, handle)
        lens = len(agent_name_keys)

        beta = [1] * lens
        ori_weight = handle.model.state_dict()
        w_locals = []
        
        # 使用列表推导式预分配空间
        for client in agent_name_keys:
            # 使用模型副本而不是深度拷贝
            model_copy = copy.deepcopy(handle.model)
            w = train.standard_train(epoch, handle.clients_data_num[client], client, handle.params,
                                     model_copy.to(device),
                                     handle.train_data[client], handle)
            w_locals.append(w)
        
        # 减少深度拷贝
        w_glob = Aggregation({k: v.clone() for k, v in ori_weight.items()}, 
                            w_locals, lens, beta, handle.params['eta'],
                            handle.params['defence_method'], handle.params)
        
        handle.model.load_state_dict(w_glob)
        acc = test.normal_test(epoch, handle.model, handle.test_data, handle.params, handle, poison=False)
        
        # 新增：记录全局准确度到数据库
        try:
            from federation_app.models import GlobalAccuracy, FederationTask
            # 获取任务对象
            task_obj = FederationTask.objects.get(task_id=handle.task_id)
            # 创建准确度记录
            GlobalAccuracy.objects.create(
                task=task_obj,
                epoch=epoch,
                accuracy=acc
            )
            handle.logger.info(f"任务 {handle.task_id} 第 {epoch} 轮全局准确度 {acc:.2f}% 已记录到数据库")  # 使用handle.logger
        except Exception as e:
            handle.logger.error(f"记录全局准确度失败: {e}")  # 使用handle.logger
        
        # 记录 epoch 时间
        handle.logger.info('Epoch {} completed in {:.2f} seconds'.format(epoch, time.time() - start_time))  # 使用handle.logger
        time.sleep(15)