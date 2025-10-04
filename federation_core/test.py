import logging
import torch
import torch.nn.functional as F
from device import device
# 移除全局logger定义，使用handle传入的logger

def normal_test(epoch, model, data_loader, params, handle, poison=False):
    """优化测试函数，减少不必要的操作"""
    # 使用当前任务的logger
    logger = handle.logger
    
    model.eval()
    test_loss = 0.0
    correct = 0
    data_num = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            batch_size = len(target)
            
            # 使用非阻塞传输
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            log_probs = model(data)
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            
            # 使用更高效的argmax
            pred = log_probs.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            data_num += batch_size
    
    test_loss /= data_num
    accuracy = 100.00 * correct / data_num
    
    logger.info('___Global_Test___g_epoch:{}  Average loss: {:.4f} Accuracy: {}/{} ({:.2f}%)'.format(
        epoch, test_loss, correct, data_num, accuracy))
    
    return accuracy