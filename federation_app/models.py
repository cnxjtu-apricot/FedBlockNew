from django.db import models
import json

class FederationTask(models.Model):
    TASK_STATUS = [
        ('running', '运行中'),
        ('paused', '已暂停'),
        ('stopped', '已停止'),
        ('completed', '已完成'),
    ]
    
    MODEL_CHOICES = [
        ('CNN', 'CNN'),
        ('r8', 'ResNet8'),
        ('r18', 'ResNet18'),
        ('r34', 'ResNet34'),
    ]
    
    DATASET_CHOICES = [
        ('MNIST', 'MNIST'),
        ('CIFAR10', 'CIFAR10'),
    ]
    
    task_id = models.CharField(max_length=50, unique=True, verbose_name="任务编号")
    task_name = models.CharField(max_length=100, verbose_name="任务名称")
    description = models.TextField(blank=True, verbose_name="任务描述")
    status = models.CharField(max_length=20, choices=TASK_STATUS, default='running', verbose_name="任务状态")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")
    current_epoch = models.IntegerField(default=0, verbose_name="当前轮次")
    total_epochs = models.IntegerField(default=100, verbose_name="总轮次")
    active_users = models.IntegerField(default=0, verbose_name="活跃用户数")
    
    model_architecture = models.CharField(max_length=20, choices=MODEL_CHOICES, default='r8', verbose_name="模型结构")
    dataset = models.CharField(max_length=20, choices=DATASET_CHOICES, default='CIFAR10', verbose_name="数据集")
    epochs = models.IntegerField(default=2000, verbose_name="训练轮数")
    reward_pool = models.DecimalField(max_digits=10, decimal_places=2, default=0.00, verbose_name="奖金池")
    
    class Meta:
        verbose_name = "联邦学习任务"
        verbose_name_plural = verbose_name
    
    def __str__(self):
        return f"{self.task_name} ({self.task_id})"
    

class TaskParticipant(models.Model):
    task = models.ForeignKey(FederationTask, on_delete=models.CASCADE, related_name='participants', verbose_name="联邦任务")
    user_id = models.IntegerField(verbose_name="用户ID")
    user_name = models.CharField(max_length=100, verbose_name="用户名")
    joined_at = models.DateTimeField(auto_now_add=True, verbose_name="加入时间")
    is_active = models.BooleanField(default=True, verbose_name="是否活跃")
    
    class Meta:
        verbose_name = "任务参与者"
        verbose_name_plural = verbose_name
        unique_together = ['task', 'user_id']
    
    def __str__(self):
        return f"用户{self.user_id} - {self.task.task_name}"

class TaskLog(models.Model):
    LOG_LEVELS = [
        ('info', '信息'),
        ('warning', '警告'),
        ('error', '错误'),
        ('success', '成功'),
    ]
    
    task = models.ForeignKey(FederationTask, on_delete=models.CASCADE, related_name='logs', verbose_name="任务")
    level = models.CharField(max_length=10, choices=LOG_LEVELS, default='info', verbose_name="日志级别")
    message = models.TextField(verbose_name="日志内容")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="记录时间")
    
    class Meta:
        verbose_name = "任务日志"
        verbose_name_plural = verbose_name
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.task.task_id} - {self.level} - {self.message[:50]}"
    

class GlobalAccuracy(models.Model):
    """全局准确度记录"""
    task = models.ForeignKey(FederationTask, on_delete=models.CASCADE, related_name='global_accuracies', verbose_name="联邦任务")
    epoch = models.IntegerField(verbose_name="训练轮次")
    accuracy = models.FloatField(verbose_name="全局准确度")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="记录时间")
    
    class Meta:
        verbose_name = "全局准确度记录"
        verbose_name_plural = verbose_name
        ordering = ['epoch']
        unique_together = ['task', 'epoch']
    
    def __str__(self):
        return f"{self.task.task_id} - 轮次{self.epoch}: {self.accuracy:.2f}%"