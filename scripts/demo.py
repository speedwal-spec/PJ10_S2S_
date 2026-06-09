"""
Gradio Web 演示界面 - A/B 测试
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ui.gradio_app import launch


def main():
    """主函数：解析参数并启动 Gradio 服务"""
    import argparse
    
    parser = argparse.ArgumentParser(description="T5 News Summarizer - Gradio Demo")
    parser.add_argument('--port', type=int, default=7860, help='服务器端口号')
    parser.add_argument('--share', action='store_true', help='创建公开分享链接')
    args = parser.parse_args()
    
    print("正在启动 MLOps Web Gateway...")
    launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
