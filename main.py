"""
T5 News Summarizer - 主入口

模块化架构：
- core/: 核心业务逻辑（模型管理）
- ui/: 用户界面（Gradio Web）
- api/: API服务（预留FastAPI扩展）

使用方式：
    python main.py                    # 启动Web界面
    python main.py --mode api         # 启动API服务（待实现）
"""
import argparse
from ui.gradio_app import launch


def main():
    """主函数：解析参数并启动相应服务"""
    parser = argparse.ArgumentParser(
        description="T5 News Summarizer - MLOps Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                      # 启动Web界面
  python main.py --port 8080          # 自定义端口
  python main.py --share              # 创建公开分享链接
        """
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        default='web',
        choices=['web', 'api'],
        help='运行模式: web (Gradio界面) 或 api (REST API服务)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='服务器端口号（默认: 7860）'
    )
    
    parser.add_argument(
        '--share',
        action='store_true',
        help='创建公开分享链接（通过Gradio临时域名）'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'web':
        print("正在启动 MLOps Web Gateway...")
        launch(server_port=args.port, share=args.share)
    elif args.mode == 'api':
        print("API服务模式敬请期待")


if __name__ == "__main__":
    main()
