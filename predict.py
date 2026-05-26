# -*- coding: utf-8 -*-
"""
T5 新闻标题/摘要生成推理脚本

【核心功能】
加载 train.py 训练好的 T5 模型，对新闻正文生成标题或摘要。
支持两种运行模式：
1. 交互模式（默认）：多轮对话，可输入编号使用预置样例或手动输入正文
2. 一次性模式：通过命令行参数传入单条文本，生成后退出

【生成长度控制】
- max_new_tokens: 解码器最多生成的 token 数量（主要控制长度）
- length_penalty: Beam Search 的长度惩罚系数
  * < 1.0: 偏好短句，更像标题
  * = 1.0: 中性
  * > 1.0: 偏好长句

【工作流程】
1. 解析命令行参数
2. 加载训练好的模型和 tokenizer
3. 加载样例数据（如果存在）
4. 根据运行模式执行：
   - 一次性模式：读取 --text 参数或 stdin，生成一次后退出
   - 交互模式：进入循环，等待用户输入，支持多种命令

【交互模式命令】
- 数字 (1-20): 使用 sample_articles_20.json 中的对应样例
- t: 多行输入模式，手动输入新闻正文
- h: 显示帮助信息
- q: 退出程序
- 其他输入: 直接作为单行正文处理

【使用示例】
# 交互模式（默认）
python predict.py

# 一次性模式
python predict.py --once --text "Breaking news..."

# 自定义生成长度
python predict.py --max_new_tokens 16 --num_beams 2

# 指定模型路径
python predict.py --ckpt /path/to/model
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ==================== 全局配置 ====================
DEFAULT_PREFIX = "summarize: "       # T5 输入前缀，与训练时保持一致
SAMPLES_NAME = "sample_articles_20.json"  # 样例文件名

# 生成更短、更像「标题/一行摘要」时优先调小 max_new_tokens；length_penalty<1 在 beam 时偏短句
PRED_MAX_NEW_TOKENS = 32             # 默认最大生成 token 数
PRED_LENGTH_PENALTY = 0.85           # 长度惩罚系数（<1 偏好短句）


def get_device() -> str:
    """
    自动检测并返回可用的计算设备
    
    【优先级】
    CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
    
    【返回值】
    - str: 设备名称 ('cuda', 'mps', 或 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ckpt_dir(p: str) -> str:
    """
    将模型检查点路径转换为绝对路径
    
    【参数】
    - p: 模型路径（可以是相对路径或绝对路径）
    
    【返回值】
    - str: 绝对路径
    """
    return p if os.path.isabs(p) else os.path.join(os.path.dirname(os.path.abspath(__file__)), p)


def _here() -> str:
    """
    获取当前脚本所在目录的绝对路径
    
    【用途】
    用于构建相对于脚本的路径，确保在不同工作目录下都能正确找到文件
    """
    return os.path.dirname(os.path.abspath(__file__))


def load_hparams(ckpt: str) -> Dict[str, Any]:
    """
    加载训练时保存的超参数配置
    
    【功能说明】
    读取 train_hparams.json，获取训练时的配置信息（如 prefix、max_source_len 等）
    这些配置需要与推理时保持一致，确保输入格式正确
    
    【参数】
    - ckpt: 模型检查点目录
    
    【返回值】
    - Dict: 超参数字典，如果文件不存在则返回空字典
    
    【重要字段】
    - prefix: 输入前缀（必须与训练时一致）
    - max_source_len: 最大输入长度
    - max_target_len: 最大输出长度
    """
    p = os.path.join(ckpt, "train_hparams.json")
    if not os.path.isfile(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sample_articles(path: str) -> Tuple[List[Dict[str, Any]], int]:
    """
    加载样例文章数据
    
    【功能说明】
    读取 sample_articles_20.json，获取预置的20条测试样例
    这些样例来自测试集，包含正文和参考摘要，便于快速测试
    
    【参数】
    - path: 样例文件路径
    
    【返回值】
    - Tuple: (items 列表, 条数)
      * items: 样例列表，每个元素包含 index, article, reference_summary
      * 条数: 样例数量
    
    【数据结构】
    items[i] = {
        'index': 1,                    # 样例编号
        'article': '新闻正文...',       # 完整正文
        'reference_summary': '参考摘要'  # 人工标注的摘要
    }
    """
    if not path or not os.path.isfile(path):
        return [], 0
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items") or []
    return items, len(items)


def main() -> None:
    """
    主函数：协调整个推理流程
    
    【工作流程】
    1. 初始化工作目录和解析参数
    2. 验证模型检查点是否存在
    3. 加载样例数据（如果存在）
    4. 加载训练配置和模型
    5. 根据运行模式执行：
       - 一次性模式：生成一次后退出
       - 交互模式：进入多轮对话循环
    
    【交互模式命令处理】
    - q/quit/exit: 退出程序
    - h/help/?: 显示帮助
    - t: 多行输入模式
    - 数字: 使用预置样例
    - 其他: 直接作为正文处理
    """
    # ==================== 1. 初始化工作目录 ====================
    here = _here()
    os.chdir(here)

    # ==================== 2. 解析命令行参数 ====================
    p = argparse.ArgumentParser(description="T5 新闻标题/摘要生成（默认可交互选样例）")
    p.add_argument("--ckpt", type=str, default="t5-news-checkpoint", help="train.py 保存的目录")
    p.add_argument(
        "--samples",
        type=str,
        default=SAMPLES_NAME,
        help="prepare_data 生成的样例 json，默认同目录下 sample_articles_20.json",
    )
    p.add_argument("--text", type=str, default="", help="与 --once 联用，生成一次后退出")
    p.add_argument(
        "--once",
        action="store_true",
        help="只生成一次后退出；可与 --text 同用；不设则进入多轮交互",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=PRED_MAX_NEW_TOKENS,
        help="解码器最多新生成几个 token（主要控长）；更短标题可试 16～24",
    )
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument(
        "--length_penalty",
        type=float,
        default=PRED_LENGTH_PENALTY,
        help="Beam 时长度惩罚：<1 偏短句、更像标题；1.0 为中性",
    )
    p.add_argument("--no_repeat_ngram", type=int, default=2, help="0 表示关闭 ngram 惩罚")
    args = p.parse_args()

    # ==================== 3. 验证模型检查点 ====================
    ck = _ckpt_dir(args.ckpt)
    if not os.path.isdir(ck) or not os.path.isfile(os.path.join(ck, "config.json")):
        print(f"未找到合法 checkpoint: {ck}\n请先运行: python train.py", file=sys.stderr)
        sys.exit(1)

    # ==================== 4. 加载样例数据 ====================
    samples_path = args.samples if os.path.isabs(args.samples) else os.path.join(here, args.samples)
    sample_items, n_sample = load_sample_articles(samples_path)

    # ==================== 5. 加载训练配置和模型 ====================
    hp = load_hparams(ck)
    # 从训练配置中获取 prefix，确保与训练时一致
    prefix = (hp.get("prefix") or DEFAULT_PREFIX) if isinstance(hp, dict) else DEFAULT_PREFIX

    # 检测计算设备
    dev = get_device()
    print("设备:", dev, flush=True)
    
    # 加载 tokenizer 和模型
    tok = AutoTokenizer.from_pretrained(ck, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(ck)
    model.to(dev)     # 将模型移动到计算设备
    model.eval()      # 切换到评估模式（关闭 Dropout 等）

    # ==================== 6. 定义单次生成函数 ====================
    def run_one(article: str) -> str:
        """
        对单篇新闻正文生成标题/摘要
        
        【工作流程】
        1. 清理输入文本
        2. 添加 prefix 并 tokenization
        3. 使用 Beam Search 生成摘要
        4. 解码并返回结果
        
        【参数】
        - article: 新闻正文
        
        【返回值】
        - str: 生成的标题/摘要
        
        【生成策略】
        - Beam Search (num_beams=4): 维护多个候选序列，选择概率最高的
        - length_penalty < 1: 偏好短句，更像标题
        - no_repeat_ngram: 避免重复的 n-gram，提高多样性
        - early_stopping: 当所有 beam 都生成 EOS 时提前停止
        """
        s = (article or "").strip()
        if not s:
            return ""
        
        # 添加 prefix，与训练时保持一致
        line = prefix + s
        
        # Tokenize 输入
        enc = tok(
            line,
            max_length=hp.get("max_source_len", 512) or 512,  # 最大输入长度
            truncation=True,   # 超过长度则截断
            return_tensors="pt",  # 返回 PyTorch tensor
        )
        enc = {k: v.to(dev) for k, v in enc.items()}  # 移动到设备
        
        # 配置生成参数
        ngram = args.no_repeat_ngram if int(args.no_repeat_ngram) > 0 else None
        gen_kw: Dict[str, Any] = {
            "max_new_tokens": int(args.max_new_tokens),  # 最多生成多少 token
            "num_beams": int(args.num_beams),            # Beam 数量
            "early_stopping": True,                      # 提前停止
        }
        if ngram is not None:
            gen_kw["no_repeat_ngram_size"] = ngram  # 禁止重复的 n-gram
        if int(args.num_beams) > 1:
            gen_kw["length_penalty"] = float(args.length_penalty)  # 长度惩罚
        
        # 执行生成（禁用梯度计算）
        with torch.no_grad():
            out = model.generate(**enc, **gen_kw)
        
        # 解码并返回
        return tok.decode(out[0], skip_special_tokens=True).strip()

    # ==================== 7. 一次性模式 ====================
    if args.once:
        # 从命令行参数或 stdin 获取输入文本
        t = (args.text or "").strip()
        if not t:
            t = (sys.stdin.readline() or "").strip()
        if not t:
            print("请提供 --text，或向 stdin 输入一行正文", file=sys.stderr)
            sys.exit(1)
        
        # 生成并打印结果
        print(run_one(t), flush=True)
        return  # 退出程序

    # ==================== 8. 交互模式 - 显示欢迎信息 ====================
    if n_sample > 0:
        # 如果加载了样例文件，显示使用说明
        print(
            f"已加载样例文件（共 {n_sample} 条）: {os.path.abspath(samples_path)}\n"
            f"  输入 1–{n_sample}：用其中一条「正文」做生成；\n"
            f"  输入 t：多行自输入一段正文，单独一行只按回车结束；\n"
            f"  输入 h：帮助；q：退出。"
        )
    else:
        # 没有样例文件时的提示
        print(
            f"未找到 {os.path.abspath(samples_path)}，请先运行 prepare_data 生成 20 条样例；\n"
            f"  仍可：输入 t 多行自输入，或 q 退出。",
        )

    # ==================== 9. 交互模式 - 主循环 ====================
    while True:
        try:
            # 读取用户输入
            s = input("文章> ").strip()
        except (EOFError, KeyboardInterrupt):
            # 处理 Ctrl+C 或 EOF
            print("\n再见。")
            break
        
        # 跳过空输入
        if not s:
            continue
        
        low = s.lower()
        
        # ---------- 命令：退出 ----------
        if low in ("q", "quit", "exit"):
            print("再见。")
            break
        
        # ---------- 命令：帮助 ----------
        if low in ("h", "help", "?"):
            if n_sample > 0:
                print(
                    f"  1–{n_sample} 使用样例；t=多行输入；h=本说明；q=退出。"
                )
            else:
                print("  t=多行输入；h=本说明；q=退出。")
            continue
        
        # ---------- 命令：多行输入 ----------
        if low == "t":
            print("多行输入正文，单独一行只按回车结束：", flush=True)
            lines: List[str] = []
            while True:
                line = input()
                # 当输入空行且已有内容时，结束输入
                if line.strip() == "" and len(lines) > 0:
                    break
                lines.append(line)
            block = "\n".join(lines).strip()
            if not block:
                print("(空，已跳过)", flush=True)
                continue
            
            # 生成并显示结果
            out = run_one(block)
            print("生成:\n" + out + "\n", flush=True)
            continue
        
        # ---------- 命令：使用样例 ----------
        if s.isdigit() and n_sample > 0:
            idx = int(s)
            if 1 <= idx <= n_sample:
                # 获取样例数据
                art = (sample_items[idx - 1].get("article") or "").strip()
                ref = (sample_items[idx - 1].get("reference_summary") or "").strip()[:200]
                
                # 显示参考摘要（前200字）
                print(f"--- 样例 {idx}（前 200 字参考摘要，仅对照）: {ref!s}…", flush=True)
                
                # 生成并显示结果
                out = run_one(art)
                print("生成:\n" + out + "\n", flush=True)
            else:
                print(f"请输入 1 到 {n_sample} 之间的数字。", file=sys.stderr)
            continue
        
        # ---------- 默认：直接作为正文处理 ----------
        out = run_one(s)
        print("生成:\n" + out + "\n", flush=True)


if __name__ == "__main__":
    main()
