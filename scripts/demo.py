#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio Web 演示界面 - A/B 测试竞技场

用法：
    python scripts/demo.py                  # 启动本地服务 (http://127.0.0.1:7860)
    python scripts/demo.py --port 8080      # 自定义端口
    python scripts/demo.py --share          # 创建公开分享链接
"""
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ui.gradio_app import launch


def main():
    """主函数：解析参数并启动 Gradio 服务"""
    import argparse
    
    parser = argparse.ArgumentParser(description="T5 News Summarizer - Gradio Demo")
    parser.add_argument('--port', type=int, default=7860, help='服务器端口号')
    parser.add_argument('--share', action='store_true', help='创建公开分享链接')
    args = parser.parse_args()
    
    print("🚀 正在启动 MLOps Web Gateway...")
    launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
