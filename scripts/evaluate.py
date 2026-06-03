#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在验证集或测试集上，对 train.py 保存的 checkpoint 计算 ROUGE-1 / ROUGE-2 / ROUGE-L。

依赖: pip install rouge-score
用法:
    python scripts/evaluate.py --ckpt checkpoints_ablation/baseline
    python scripts/evaluate.py --ckpt checkpoints_ablation/baseline --split test --max_samples 0
"""
import os
import sys
import json
import argparse
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.report_generator import generate_all_reports


def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _abs(p: str) -> str:
    """将相对路径解析为基于项目根目录的绝对路径"""
    if os.path.isabs(p):
        return p
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return os.path.join(project_root, p)


def load_manifest(path: str) -> Dict[str, Any]:
    """加载数据清单"""
    p = _abs(path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"未找到 {p}，请先运行 prepare_data.py")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_raw_dataset(manifest: Dict[str, Any]):
    """从本地缓存加载数据集"""
    name = (manifest.get("dataset_name") or "cnn_dailymail").strip()
    cache_dir = manifest.get("cache_dir")
    if not cache_dir or not os.path.isdir(cache_dir):
        raise FileNotFoundError(
            f"manifest 中 cache_dir 无效: {cache_dir!r}"
        )
    if name.lower() == "xsum":
        return load_dataset("xsum", cache_dir=cache_dir)
    cfg = manifest.get("dataset_config") or "3.0.0"
    return load_dataset(name, cfg, cache_dir=cache_dir)


def get_device() -> str:
    """检测设备类型"""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_rouge(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """计算 ROUGE 指标"""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    
    for p, g in zip(preds, refs):
        p = (p or "").strip()
        g = (g or "").strip()
        if not g:
            continue
        sc = scorer.score(g, p)
        r1.append(sc["rouge1"].fmeasure)
        r2.append(sc["rouge2"].fmeasure)
        rl.append(sc["rougeL"].fmeasure)
    
    n = len(r1)
    if n == 0:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "n": 0}
    
    return {
        "rouge1": float(np.mean(r1)),
        "rouge2": float(np.mean(r2)),
        "rougeL": float(np.mean(rl)),
        "n": n,
    }


def main() -> None:
    """主函数：执行 ROUGE 评测"""
    # ✅ 确定项目根目录（不改变工作目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    parser = argparse.ArgumentParser(description="评测模型：ROUGE-1/2/L")
    parser.add_argument("--ckpt", type=str, default="checkpoints_ablation/baseline", 
                       help="checkpoint 目录")
    parser.add_argument("--manifest", type=str, default="data_manifest.json")
    parser.add_argument("--split", type=str, default="validation",
                       choices=["validation", "test"])
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="最多评测条数，0=全量")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--length_penalty", type=float, default=0.85)
    parser.add_argument("--no_repeat_ngram", type=int, default=2)
    parser.add_argument("--output_json", type=str, default="",
                       help="输出 JSON 路径，空则只打印")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from rouge_score import rouge_scorer  # noqa: F401
    except ImportError:
        print("请先安装: pip install rouge-score", file=sys.stderr)
        sys.exit(1)

    ck = _abs(args.ckpt)
    if not os.path.isdir(ck) or not os.path.isfile(os.path.join(ck, "config.json")):
        print(f"未找到 checkpoint: {ck}", file=sys.stderr)
        sys.exit(1)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest = load_manifest(args.manifest)
    text_col = manifest["text_column"]
    summary_col = manifest["summary_column"]
    
    print("正在从本地 cache 加载数据集…", flush=True)
    raw = _load_raw_dataset(manifest)
    print("数据集已载入。", flush=True)
    
    if args.split not in raw:
        print(f"数据集中无 split: {args.split!r}", file=sys.stderr)
        sys.exit(1)
    
    ds = raw[args.split]
    n_all = len(ds)
    n_take = min(int(args.max_samples), n_all) if int(args.max_samples) > 0 else n_all
    
    if n_take < n_all:
        ds = ds.shuffle(seed=args.seed).select(range(n_take))
    
    device = get_device()
    print(f"设备: {device} | 评测样本数: {n_take}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(ck, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(ck)
    model.to(device)
    model.eval()

    hp_path = os.path.join(ck, "train_hparams.json")
    hparams = {}
    if os.path.isfile(hp_path):
        with open(hp_path, "r", encoding="utf-8") as f:
            hparams = json.load(f)
    
    prefix = hparams.get("prefix", "summarize: ")
    max_source_len = hparams.get("max_source_len", 512)

    preds, refs = [], []
    batch_size = args.batch_size
    
    print("开始生成预测…", flush=True)
    with torch.no_grad():
        for i in range(0, n_take, batch_size):
            sl = ds.select(range(i, min(i + batch_size, n_take)))
            ins = [prefix + str(t) for t in sl[text_col]]
            
            enc = tokenizer(
                ins,
                max_length=max_source_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram,
                early_stopping=True,
            )
            
            preds.extend(tokenizer.batch_decode(gen.cpu(), skip_special_tokens=True))
            refs.extend([str(s) for s in sl[summary_col]])
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  已处理 {min(i + batch_size, n_take)}/{n_take} 条", flush=True)

    rouge_scores = compute_rouge(preds, refs)
    
    print("\n" + "="*60)
    print("📊 ROUGE 评测结果")
    print("="*60)
    print(f"  ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"  ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"  ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print(f"  评测样本数: {rouge_scores['n']}")
    print("="*60)

    # 输出示例
    print("\n📝 生成样例（前3条）:")
    for k in range(min(3, len(preds))):
        print(f"\n[样例 {k+1}]")
        print(f"参考: {refs[k][:150]}...")
        print(f"生成: {preds[k]}")

    # 写入 JSON（保留原有功能）
    if args.output_json:
        out_path = args.output_json if os.path.isabs(args.output_json) else _abs(args.output_json)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "exp_id": os.path.basename(args.ckpt),
                "split": args.split,
                "n_samples": rouge_scores["n"],
                "rouge1": round(rouge_scores["rouge1"], 4),
                "rouge2": round(rouge_scores["rouge2"], 4),
                "rougeL": round(rouge_scores["rougeL"], 4),
            }, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 结果已写入: {out_path}")
    
    # ✅ 自动生成完整报告（JSON + Markdown + 文本）
    print("\n📝 正在生成评测报告...")
    exp_id = os.path.basename(args.ckpt)
    generated_files = generate_all_reports(
        rouge_scores=rouge_scores,
        preds=preds,
        refs=refs,
        exp_id=exp_id,
        split=args.split,
        ckpt_path=args.ckpt,
        project_root=project_root,
    )
    
    print("\n" + "="*60)
    print("📄 生成的报告文件:")
    print("="*60)
    print(f"  JSON 结果:   {generated_files['json']}")
    print(f"  Markdown:    {generated_files['markdown']} (符合评测要求)")
    print(f"  文本报告:    {generated_files['text']}")
    print("="*60)


if __name__ == "__main__":
    main()
