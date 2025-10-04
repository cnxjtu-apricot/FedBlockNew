#!/usr/bin/env python
"""Django management script."""
import os
import sys
import logging  # 添加logging导入

def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'federation_platform.settings')
    
    # 关键修改：禁用根logger的默认处理器，避免干扰任务日志
    logging.getLogger().handlers = []
    logging.getLogger().propagate = False
    
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django."
        ) from exc
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()