#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻摘要 / 标题生成任务 — 数据准备（独立脚本，不依赖项目其它源码）
从 Hugging Face 下载 cnn_dailymail 或 xsum 到本地缓存，并生成：

- data_manifest.json：字段与 cache 路径约定
- sample_articles_20.json：从测试集抽取 20 条（英文）正文+参考摘要，供 predict 选编号试跑
"""

import argparse
import json
import os

from datasets import load_dataset

# 可选：国内镜像，便于在部分网络环境下加速访问 Hugging Face
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def get_text_summary_columns(dataset_name: str):
    """返回（正文列名, 摘要/标题列名）。"""
    if dataset_name.lower() == "xsum":
        return "document", "summary"
    # cnn_dailymail
    return "article", "highlights"


def main():
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

    os.makedirs(args.cache_dir, exist_ok=True)
    name = args.dataset.strip()

    print(f"正在下载数据集: {name} -> {args.cache_dir}")

    if name.lower() == "xsum":
        ds = load_dataset("xsum", cache_dir=args.cache_dir)
        cfg_used = None
    else:
        ds = load_dataset(name, args.dataset_config, cache_dir=args.cache_dir)
        cfg_used = args.dataset_config

    text_col, summary_col = get_text_summary_columns(name)

    train_n = len(ds["train"])
    val_n = len(ds["validation"])
    test_n = len(ds["test"])

    manifest = {
        "dataset_name": name,
        "dataset_config": cfg_used,
        "cache_dir": os.path.abspath(args.cache_dir),
        "text_column": text_col,
        "summary_column": summary_col,
        "splits": {
            "train": train_n,
            "validation": val_n,
            "test": test_n,
        },
        "note": "cnn_dailymail 的 highlights 列为多句摘要/要点，可作标题生成监督；T5 常用输入前缀 summarize: ",
    }

    out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_manifest.json")
    base_dir = os.path.dirname(out_json)

    # 20 条测试样例（英文正文+参考摘要，便于人工试 predict，无需自撰长文）
    sample_name = "sample_articles_20.json"
    sample_path = os.path.join(base_dir, sample_name)
    test_split = ds["test"]
    n_take = min(20, len(test_split))
    test_shuf = test_split.shuffle(seed=42).select(range(n_take))
    items = []
    for i in range(n_take):
        row = test_shuf[i]
        art = str(row[text_col])
        ref = str(row[summary_col])
        items.append(
            {
                "index": i + 1,
                "article": art,
                "reference_summary": ref,
            }
        )
    sample_payload = {
        "dataset": name,
        "text_column": text_col,
        "summary_column": summary_col,
        "source_split": "test",
        "sample_seed": 42,
        "count": n_take,
        "items": items,
        "note": "由 prepare_data 从测试集固定随机顺序抽取，供 predict 交互选 1-20 条试跑；参考摘要可对照人工主观质量。",
    }
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample_payload, f, ensure_ascii=False, indent=2)

    manifest["sample_articles_file"] = os.path.basename(sample_path)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("下载与元数据写入完成。")
    print(f"  train={train_n}, validation={val_n}, test={test_n}")
    print(f"  正文列: {text_col}, 摘要列: {summary_col}")
    print(f"  已保存: {out_json}")
    print(f"  测试样例: {n_take} 条 -> {sample_path}")


if __name__ == "__main__":
    main()
