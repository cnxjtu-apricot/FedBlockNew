from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .services import task_manager
from .models import FederationTask, TaskLog, TaskParticipant, GlobalAccuracy

import logging

logger = logging.getLogger("logger")

def dashboard(request):
    """主控制面板"""
    tasks_status = task_manager.get_all_tasks_status()
    
    # 获取任务日志
    recent_logs = TaskLog.objects.select_related('task').order_by('-created_at')[:20]
    
    context = {
        'tasks_status': tasks_status,
        'recent_logs': recent_logs,
    }
    return render(request, 'federation_app/dashboard.html', context)

# views.py - 修改 create_task 视图
@csrf_exempt
def create_task(request):
    """创建联邦任务"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            task_id = data.get('task_id')
            task_name = data.get('task_name')
            description = data.get('description', '')
            
            # 新增参数
            model_architecture = data.get('model_architecture', 'r8')
            dataset = data.get('dataset', 'CIFAR10')
            epochs = int(data.get('epochs', 2000))
            reward_pool = float(data.get('reward_pool', 0.00))
            
            if not task_id or not task_name:
                return JsonResponse({'success': False, 'message': '任务编号和名称不能为空'})
            
            # 参数验证
            if model_architecture not in ['CNN', 'r8', 'r18', 'r34']:
                return JsonResponse({'success': False, 'message': '不支持的模型结构'})
            
            if dataset not in ['MNIST', 'CIFAR10']:
                return JsonResponse({'success': False, 'message': '不支持的数据集'})
            
            if epochs <= 0:
                return JsonResponse({'success': False, 'message': '训练轮数必须大于0'})
            
            if reward_pool < 0:
                return JsonResponse({'success': False, 'message': '奖金池不能为负数'})
            
            task_obj = task_manager.create_task(
                task_id=task_id,
                task_name=task_name,
                description=description,
                model_architecture=model_architecture,
                dataset=dataset,
                epochs=epochs,
                reward_pool=reward_pool
            )
            task_manager.start_task(task_id)
            
            return JsonResponse({
                'success': True, 
                'message': f'联邦任务 {task_name} 创建并启动成功',
                'task_id': task_id
            })
            
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})
    
    return JsonResponse({'success': False, 'message': '仅支持POST请求'})

@csrf_exempt
def join_task(request):
    """加入联邦任务"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            task_id = data.get('task_id')
            user_id = data.get('user_id')
            user_name = data.get('user_name', f'用户{user_id}')
            
            if not task_id or user_id is None:
                return JsonResponse({'success': False, 'message': '任务编号和用户ID不能为空'})
            
            success = task_manager.add_user_to_task(task_id, int(user_id), user_name)
            
            if success:
                return JsonResponse({
                    'success': True, 
                    'message': f'用户 {user_name} 成功加入联邦任务 {task_id}'
                })
            else:
                return JsonResponse({
                    'success': False, 
                    'message': f'用户 {user_name} 加入联邦任务失败'
                })
                
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})
    
    return JsonResponse({'success': False, 'message': '仅支持POST请求'})

@csrf_exempt
def leave_task(request):
    """退出联邦任务"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            task_id = data.get('task_id')
            user_id = data.get('user_id')
            
            if not task_id or user_id is None:
                return JsonResponse({'success': False, 'message': '任务编号和用户ID不能为空'})
            
            success = task_manager.remove_user_from_task(task_id, int(user_id))
            
            if success:
                return JsonResponse({
                    'success': True, 
                    'message': f'用户 {user_id} 成功退出联邦任务 {task_id}'
                })
            else:
                return JsonResponse({
                    'success': False, 
                    'message': f'用户 {user_id} 退出联邦任务失败'
                })
                
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})
    
    return JsonResponse({'success': False, 'message': '仅支持POST请求'})

@csrf_exempt
def delete_task(request):
    """删除联邦任务"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            task_id = data.get('task_id')
            
            if not task_id:
                return JsonResponse({'success': False, 'message': '任务编号不能为空'})
            
            # 检查任务是否存在
            try:
                task_obj = FederationTask.objects.get(task_id=task_id)
            except FederationTask.DoesNotExist:
                return JsonResponse({'success': False, 'message': f"任务 {task_id} 不存在"})
            
            # 检查任务是否正在运行
            if task_id in task_manager.tasks:
                task_data = task_manager.tasks[task_id]
                if task_data['thread'] and task_data['thread'].is_alive():
                    return JsonResponse({
                        'success': False, 
                        'message': f'任务 {task_id} 正在运行中，无法删除。请先停止任务。'
                    })
            
            # 删除任务相关的所有数据
            # 1. 从任务管理器中移除（如果存在）
            if task_id in task_manager.tasks:
                del task_manager.tasks[task_id]
            
            # 2. 删除数据库记录（由于外键关联，相关记录也会被删除）
            task_name = task_obj.task_name
            task_obj.delete()
            
            logger.info(f"任务 {task_id} 及相关数据已删除")
            
            return JsonResponse({
                'success': True, 
                'message': f'任务 {task_name} 已成功删除'
            })
                
        except Exception as e:
            logger.error(f"删除任务失败: {e}")
            return JsonResponse({'success': False, 'message': str(e)})
    
    return JsonResponse({'success': False, 'message': '仅支持POST请求'})

@csrf_exempt
def clear_logs(request):
    """清空所有日志"""
    if request.method == 'POST':
        try:
            # 删除所有日志记录
            count = TaskLog.objects.all().delete()[0]
            
            logger.info(f"已清空 {count} 条日志记录")
            
            return JsonResponse({
                'success': True, 
                'message': f'已清空 {count} 条日志记录'
            })
                
        except Exception as e:
            logger.error(f"清空日志失败: {e}")
            return JsonResponse({'success': False, 'message': str(e)})
    
    return JsonResponse({'success': False, 'message': '仅支持POST请求'})

def get_task_status(request, task_id):
    """获取任务状态"""
    try:
        status = task_manager.get_task_status(task_id)
        return JsonResponse({'success': True, 'status': status})
    except Exception as e:
        return JsonResponse({'success': False, 'message': str(e)})

def get_all_status(request):
    """获取所有任务状态"""
    try:
        status_list = task_manager.get_all_tasks_status()
        return JsonResponse({'success': True, 'tasks': status_list})
    except Exception as e:
        return JsonResponse({'success': False, 'message': str(e)})
    
@csrf_exempt
def get_accuracy_history(request):
    """获取任务准确度历史"""
    if request.method == 'GET':
        task_id = request.GET.get('task_id')
        
        if not task_id:
            return JsonResponse({'success': False, 'message': '任务编号不能为空'})
        
        try:
            result = task_manager.get_task_accuracy_history(task_id)
            return JsonResponse(result)
        except Exception as e:
            logger.error(f"获取准确度历史API错误: {e}")
            return JsonResponse({'success': False, 'message': str(e)})
    
    return JsonResponse({'success': False, 'message': '仅支持GET请求'})

@csrf_exempt
def get_logs(request):
    """获取系统日志"""
    if request.method == 'GET':
        try:
            # 获取最近的20条日志
            recent_logs = TaskLog.objects.select_related('task').order_by('-created_at')[:20]
            
            logs_data = []
            for log in recent_logs:
                logs_data.append({
                    'task_id': log.task.task_id,
                    'level': log.level,
                    'level_display': log.get_level_display(),
                    'message': log.message,
                    'created_at': log.created_at.isoformat()
                })
            
            return JsonResponse({
                'success': True,
                'logs': logs_data
            })
        except Exception as e:
            logger.error(f"获取系统日志失败: {e}")
            return JsonResponse({'success': False, 'message': str(e)})
    
    return JsonResponse({'success': False, 'message': '仅支持GET请求'})