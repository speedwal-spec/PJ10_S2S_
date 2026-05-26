#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻摘要 / 标题生成任务 — 数据准备脚本

【核心功能】
本脚本负责从 Hugging Face Hub 下载新闻摘要数据集（cnn_dailymail 或 xsum），
并将其缓存到本地，同时生成两个关键文件：

1. data_manifest.json：数据清单文件，记录数据集的元信息（路径、列名、样本数等）
   - 供 train.py 和 evaluate_rouge.py 读取，确保训练和评估使用相同的数据配置
   
2. sample_articles_20.json：从测试集中抽取的20条样例数据
   - 包含英文正文(article)和参考摘要(reference_summary)
   - 供 predict.py 在交互模式下快速测试，无需手动输入长文本

【工作流程】
1. 解析命令行参数（数据集名称、配置版本、缓存目录）
2. 使用 datasets.load_dataset() 从 Hugging Face 下载并缓存数据集
3. 提取数据集的列名信息（正文字段、摘要字段）
4. 统计各划分（train/validation/test）的样本数量
5. 生成 data_manifest.json 元数据文件
6. 从测试集随机抽取20条样例，保存为 sample_articles_20.json

【依赖说明】
- datasets: Hugging Face 提供的数据集加载库，支持自动下载、缓存和高效读取
- 首次运行需联网下载数据集（约3GB），后续运行直接从本地缓存加载

【使用示例】
# 使用默认数据集 cnn_dailymail
python prepare_data.py

# 使用 xsum 数据集
python prepare_data.py --dataset xsum --cache_dir ./data_cache

# 指定自定义缓存目录
python prepare_data.py --cache_dir /path/to/cache
"""

import argparse
import json
import os

from datasets import load_dataset

# 设置 Hugging Face 镜像地址（可选）
# 国内网络环境下可加速访问，若网络良好可注释此行
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def get_text_summary_columns(dataset_name: str):
    """
    根据数据集名称返回对应的正文字段名和摘要字段名
    
    【设计原因】
    不同数据集的字段命名不同，需要统一映射：
    - cnn_dailymail: 使用 'article'（正文）和 'highlights'（要点摘要）
    - xsum: 使用 'document'（文档）和 'summary'（摘要）
    
    【参数说明】
    - dataset_name: 数据集名称，如 'cnn_dailymail' 或 'xsum'
    
    【返回值】
    - tuple: (text_column, summary_column)，即正文字段名和摘要字段名
    
    【使用场景】
    在数据预处理时，需要根据字段名从数据集中提取正确的列
    """
    if dataset_name.lower() == "xsum":
        # XSum 数据集使用 document/summary 作为字段名
        return "document", "summary"
    # CNN/DailyMail 数据集使用 article/highlights 作为字段名
    # highlights 是多句摘要/要点，适合作为标题生成的监督信号
    return "article", "highlights"


def main():
    """
    主函数：执行完整的数据准备流程
    
    【核心步骤】
    1. 解析命令行参数
    2. 创建缓存目录
    3. 加载数据集（自动下载或从缓存读取）
    4. 获取数据集的字段名配置
    5. 统计各划分的样本数量
    6. 生成 data_manifest.json（数据元信息清单）
    7. 从测试集抽取20条样例，生成 sample_articles_20.json
    8. 输出完成信息和数据统计
    
    【关键技术点】
    - load_dataset(): Hugging Face datasets 库的核心函数，支持：
      * 自动下载数据集到本地缓存
      * 支持断点续传和网络中断恢复
      * 使用 Arrow 格式高效存储和读取大规模数据
    
    - 数据集划分：
      * train: 训练集，用于模型参数学习
      * validation: 验证集，用于超参数调优和早停
      * test: 测试集，用于最终性能评估
    
    【注意事项】
    - 首次运行需要较长时间下载数据集（取决于网络速度）
    - 缓存目录会占用较大磁盘空间（CNN/DailyMail 约3GB）
    - 样本抽取使用固定种子(seed=42)，保证可复现性
    """
    # ==================== 1. 解析命令行参数 ====================
    parser = argparse.ArgumentParser(description="下载新闻摘要数据集到本地缓存")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cnn_dailymail",
        help="数据集名：cnn_dailymail 或 xsum",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="3.0.0",
        help="cnn_dailymail 的配置版本；使用 xsum 时忽略",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./data_cache",
        help="数据集缓存目录",
    )
    args = parser.parse_args()

    # ==================== 2. 创建缓存目录 ====================
    # exist_ok=True 确保目录已存在时不会报错
    os.makedirs(args.cache_dir, exist_ok=True)
    name = args.dataset.strip()

    # ==================== 3. 加载数据集 ====================
    print(f"正在下载数据集: {name} -> {args.cache_dir}")

    # 根据数据集类型选择不同的加载方式
    if name.lower() == "xsum":
        # XSum 数据集无需配置版本参数
        ds = load_dataset("xsum", cache_dir=args.cache_dir)
        cfg_used = None
    else:
        # CNN/DailyMail 需要指定配置版本（3.0.0是常用版本）
        ds = load_dataset(name, args.dataset_config, cache_dir=args.cache_dir)
        cfg_used = args.dataset_config

    # ==================== 4. 获取字段名配置 ====================
    text_col, summary_col = get_text_summary_columns(name)

    # ==================== 5. 统计各划分样本数量 ====================
    train_n = len(ds["train"])
    val_n = len(ds["validation"])
    test_n = len(ds["test"])

    # ==================== 6. 构建数据清单 manifest ====================
    manifest = {
        "dataset_name": name,              # 数据集名称
        "dataset_config": cfg_used,         # 配置版本（xsum为None）
        "cache_dir": os.path.abspath(args.cache_dir),  # 缓存目录的绝对路径
        "text_column": text_col,            # 正文字段名
        "summary_column": summary_col,      # 摘要字段名
        "splits": {                         # 各划分的样本数量
            "train": train_n,
            "validation": val_n,
            "test": test_n,
        },
        "note": "cnn_dailymail 的 highlights 列为多句摘要/要点，可作标题生成监督；T5 常用输入前缀 summarize: ",
    }

    # 保存 manifest 到 JSON 文件
    out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_manifest.json")
    base_dir = os.path.dirname(out_json)

    # ==================== 7. 抽取20条测试样例 ====================
    # 目的：为 predict.py 提供快速测试的样例数据，避免手动输入长文本
    sample_name = "sample_articles_20.json"
    sample_path = os.path.join(base_dir, sample_name)
    
    # 从测试集中抽取样例
    test_split = ds["test"]
    n_take = min(20, len(test_split))  # 最多取20条，防止测试集过小
    
    # 使用固定种子shuffle，保证每次运行抽取的样例一致（可复现性）
    test_shuf = test_split.shuffle(seed=42).select(range(n_take))
    
    # 构建样例列表
    items = []
    for i in range(n_take):
        row = test_shuf[i]
        art = str(row[text_col])      # 提取正文
        ref = str(row[summary_col])   # 提取参考摘要
        items.append(
            {
                "index": i + 1,                    # 样例编号（从1开始）
                "article": art,                     # 新闻正文
                "reference_summary": ref,           # 参考摘要（用于对比评估）
            }
        )
    
    # 构建样例文件的完整数据结构
    sample_payload = {
        "dataset": name,                   # 数据集名称
        "text_column": text_col,           # 正文字段名
        "summary_column": summary_col,     # 摘要字段名
        "source_split": "test",            # 样例来源划分
        "sample_seed": 42,                 # 随机种子
        "count": n_take,                   # 样例数量
        "items": items,                    # 样例列表
        "note": "由 prepare_data 从测试集固定随机顺序抽取，供 predict 交互选 1-20 条试跑；参考摘要可对照人工主观质量。",
    }
    
    # 保存样例文件
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample_payload, f, ensure_ascii=False, indent=2)

    # 将样例文件名添加到 manifest 中
    manifest["sample_articles_file"] = os.path.basename(sample_path)
    
    # 保存 manifest 文件
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ==================== 8. 输出完成信息 ====================
    print("下载与元数据写入完成。")
    print(f"  train={train_n}, validation={val_n}, test={test_n}")
    print(f"  正文列: {text_col}, 摘要列: {summary_col}")
    print(f"  已保存: {out_json}")
    print(f"  测试样例: {n_take} 条 -> {sample_path}")


if __name__ == "__main__":
    main()
