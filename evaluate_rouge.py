# -*- coding: utf-8 -*-
"""
在验证集或测试集上，对 train.py 保存的 checkpoint 计算 ROUGE-1 / ROUGE-2 / ROUGE-L（F1，与 rouge-score 一致）。

依赖: pip install rouge-score
用法:
  python evaluate_rouge.py --ckpt t5-news-checkpoint
  python evaluate_rouge.py --ckpt t5-news-checkpoint --split test --max_samples 0
  最后一类为全量评测（可能很慢）。默认 max_samples=1000 时，在划分内按 --seed（默认 42）shuffle 后取子集，结果可复现。
"""
# =============================================================================
# 默认参数（现在由 configs/default.yaml 统一管理）
# 运行方式:
#   python evaluate_rouge.py --ckpt checkpoints_ablation/Exp-01_Baseline
# =============================================================================
MANIFEST = "data_manifest.json"
CKPT_DIR = "t5-news-checkpoint"
SPLIT = "validation"
# 最多评测 1000 条；全量用 --max_samples 0
MAX_SAMPLES = 1000
BATCH_SIZE = 4
MAX_NEW_TOKENS = 32
NUM_BEAMS = 4
LENGTH_PENALTY = 0.85
NGRAM_PENALTY = 2
OUT_JSON = ""
# =============================================================================


import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from configs.config_manager import load_config


def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_here(), p)


def load_manifest(path: str) -> Dict[str, Any]:
    p = _abs(path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"未找到 {p}，请先运行 prepare_data.py")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_hparams(ckpt: str) -> Dict[str, Any]:
    p = os.path.join(ckpt, "train_hparams.json")
    if not os.path.isfile(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_raw_dataset(manifest: Dict[str, Any]):
    name = (manifest.get("dataset_name") or "cnn_dailymail").strip()
    cache_dir = manifest.get("cache_dir")
    if not cache_dir or not os.path.isdir(cache_dir):
        raise FileNotFoundError(
            f"manifest 中 cache_dir 无效: {cache_dir!r}，请重新执行 prepare_data.py"
        )
    if name.lower() == "xsum":
        return load_dataset("xsum", cache_dir=cache_dir)
    cfg = manifest.get("dataset_config") or "3.0.0"
    return load_dataset(name, cfg, cache_dir=cache_dir)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_rouge_batch(preds: List[str], refs: List[str]) -> Dict[str, float]:
    from rouge_score import rouge_scorer

    s = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for p, g in zip(preds, refs):
        p = (p or "").strip()
        g = (g or "").strip()
        if not g:
            continue
        sc = s.score(g, p)
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
    os.chdir(_here())
    p = argparse.ArgumentParser(description="评测 train 保存的模型：ROUGE-1/2/L")
    p.add_argument("--ckpt", type=str, default=CKPT_DIR, help="train.py 输出目录")
    p.add_argument("--manifest", type=str, default=MANIFEST)
    p.add_argument(
        "--split",
        type=str,
        default=SPLIT,
        choices=["validation", "test"],
        help="使用哪个划分求 ROUGE",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=MAX_SAMPLES,
        help="最多评多少条：若小于当前划分总条数，则先 shuffle(--seed) 再取前 N 条（可复现随机子集）；0=全量（CPU 上可能极慢）",
    )
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--num_beams", type=int, default=NUM_BEAMS)
    p.add_argument("--length_penalty", type=float, default=LENGTH_PENALTY)
    p.add_argument("--no_repeat_ngram", type=int, default=NGRAM_PENALTY)
    p.add_argument(
        "--output_json",
        type=str,
        default=OUT_JSON,
        help="将指标写入该 JSON 路径；空则只打印",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="numpy/torch 种子；子集抽样时亦用于对划分 shuffle，保证同一命令可复现",
    )
    p.add_argument("--config", type=str, default=None,
                   help="配置文件路径（如 configs/default.yaml），覆盖评测默认值")
    args = p.parse_args()

    # 如果指定了 --config，从 YAML 加载评测默认值
    if args.config:
        cfg = load_config(args.config)
        if args.batch_size == BATCH_SIZE:
            args.batch_size = cfg.evaluation.batch_size
        if args.max_samples == MAX_SAMPLES:
            args.max_samples = cfg.evaluation.max_samples
        if args.max_new_tokens == MAX_NEW_TOKENS:
            args.max_new_tokens = cfg.inference.max_new_tokens
        if args.num_beams == NUM_BEAMS:
            args.num_beams = cfg.inference.num_beams
        if args.length_penalty == LENGTH_PENALTY:
            args.length_penalty = cfg.inference.length_penalty
        if args.no_repeat_ngram == NGRAM_PENALTY:
            args.no_repeat_ngram = cfg.inference.no_repeat_ngram

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
    print(
        "正在从本地 cache 加载数据集（Arrow 大表首次映射可能 1～数分钟，并非死机）…",
        flush=True,
    )
    raw = _load_raw_dataset(manifest)
    print("数据集已载入。", flush=True)
    if args.split not in raw:
        print(f"数据集中无 split: {args.split!r}，可用: {list(raw.keys())}", file=sys.stderr)
        sys.exit(1)
    ds = raw[args.split]
    n_all = len(ds)
    n_take = n_all
    if int(args.max_samples) > 0:
        n_take = min(int(args.max_samples), n_all)
    else:
        if n_all > 2000 and get_device() == "cpu":
            print(
                f"提示: 全量 {n_all} 条、CPU+beam 生成可能需数小时；可中断后用 "
                f"--max_samples 500 先测。",
                file=sys.stderr,
                flush=True,
            )
    if n_take < n_all:
        ds = ds.shuffle(seed=int(args.seed)).select(range(n_take))
        sampling_note = f"可复现随机子集（shuffle, seed={args.seed}）{n_take} 条"
    else:
        sampling_note = f"全量 {n_take} 条"

    hp = load_hparams(ck)
    prefix = (hp.get("prefix") or "summarize: ") if isinstance(hp, dict) else "summarize: "
    max_src = int(hp.get("max_source_len", 512) or 512)

    dev = get_device()
    print(
        f"设备: {dev} | split={args.split} | 本评价样本数: {n_take} / 划分共 {n_all} | {sampling_note}",
        flush=True,
    )
    print("正在加载 tokenizer…", flush=True)
    tok = AutoTokenizer.from_pretrained(ck, use_fast=True)
    print("正在加载模型权重（若从网络路径首次拉取会较慢；本地 ckpt 一般数秒至一两分钟）…", flush=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(ck)
    model.to(dev)
    model.eval()
    print("开始生成 + ROUGE 汇总（下方为进度条）", flush=True)

    all_preds: List[str] = []
    all_refs: List[str] = []
    bs = max(1, int(args.batch_size))
    ngram = int(args.no_repeat_ngram) if int(args.no_repeat_ngram) > 0 else None

    n_batches = (n_take + bs - 1) // bs
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None  # type: ignore

    with torch.no_grad():
        it = range(0, n_take, bs)
        if tqdm is not None and n_batches > 1:
            it = tqdm(it, total=n_batches, desc="generate", unit="batch", file=sys.stdout)
        for s in it:
            sl = ds.select(range(s, min(s + bs, n_take)))
            articles = [str(t) for t in sl[text_col]]
            refs = [str(t) for t in sl[summary_col]]
            ins = [prefix + a for a in articles]
            enc = tok(
                ins,
                max_length=max_src,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(dev) for k, v in enc.items()}
            gen_kw: Dict[str, Any] = {
                "max_new_tokens": int(args.max_new_tokens),
                "num_beams": int(args.num_beams),
                "early_stopping": True,
            }
            if ngram is not None:
                gen_kw["no_repeat_ngram_size"] = ngram
            if int(args.num_beams) > 1:
                gen_kw["length_penalty"] = float(args.length_penalty)
            out = model.generate(**enc, **gen_kw)
            decoded = tok.batch_decode(out, skip_special_tokens=True)
            all_preds.extend([x.strip() for x in decoded])
            all_refs.extend(refs)

    print("正在计算 ROUGE-1/2/L…", flush=True)
    m = compute_rouge_batch(all_preds, all_refs)
    n_used = m.pop("n", len(all_refs))

    print("\n======== ROUGE (F1, 英文 stemmer) ========")
    print(f"  ROUGE-1: {m['rouge1']:.4f}")
    print(f"  ROUGE-2: {m['rouge2']:.4f}")
    print(f"  ROUGE-L: {m['rougeL']:.4f}")
    print(f"  条数: {n_used}")
    print("========================================\n", flush=True)

    payload = {
        "rouge1": m["rouge1"],
        "rouge2": m["rouge2"],
        "rougeL": m["rougeL"],
        "n_samples": n_used,
        "split": args.split,
        "seed": int(args.seed),
        "sampling": sampling_note,
        "ckpt": os.path.abspath(ck),
        "gen": {
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "length_penalty": args.length_penalty,
            "no_repeat_ngram": args.no_repeat_ngram,
        },
    }
    if args.output_json:
        out_path = _abs(args.output_json)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("已写入:", out_path, flush=True)


if __name__ == "__main__":
    main()
