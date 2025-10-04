from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('api/tasks/create/', views.create_task, name='create_task'),
    path('api/tasks/join/', views.join_task, name='join_task'),
    path('api/tasks/leave/', views.leave_task, name='leave_task'),
    path('api/tasks/delete/', views.delete_task, name='delete_task'),
    path('api/tasks/clear-logs/', views.clear_logs, name='clear_logs'),
    path('api/tasks/status/', views.get_all_status, name='get_all_status'),
    path('api/tasks/<str:task_id>/status/', views.get_task_status, name='get_task_status'),
    path('api/tasks/accuracy-history/', views.get_accuracy_history, name='get_accuracy_history'),
    path('api/tasks/logs/', views.get_logs, name='get_logs'),
]