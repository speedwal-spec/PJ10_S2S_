# -*- coding: utf-8 -*-
"""
加载 train.py 保存的 T5 权重，对新闻正文生成标题/摘要。

生成长度主要由 --max_new_tokens 控制（解码步数上限）；Beam 下 --length_penalty<1 会偏短句、更像标题。
训练侧由 train.py 的 MAX_TARGET_LENGTH 控制监督截断，需更短风格时应重训或调小该值。

- 默认：交互式多轮；可输入 1–20 使用 prepare_data 生成的样例，或 t 多行自输入，q 退出。
- 一次性：python predict.py --once --text "..." 等。
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from configs.config_manager import load_config

DEFAULT_PREFIX = "summarize: "
SAMPLES_NAME = "sample_articles_20.json"

# 生成更短、更像「标题/一行摘要」时优先调小 max_new_tokens；length_penalty<1 在 beam 时偏短句
# 提示：这些默认值现在统一在 configs/default.yaml 的 inference 字段中定义
PRED_MAX_NEW_TOKENS = 32
PRED_LENGTH_PENALTY = 0.85


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ckpt_dir(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(os.path.dirname(os.path.abspath(__file__)), p)


def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def load_hparams(ckpt: str) -> Dict[str, Any]:
    p = os.path.join(ckpt, "train_hparams.json")
    if not os.path.isfile(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sample_articles(path: str) -> Tuple[List[Dict[str, Any]], int]:
    """返回 (items 列表, 条数)。"""
    if not path or not os.path.isfile(path):
        return [], 0
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items") or []
    return items, len(items)


def main() -> None:
    here = _here()
    os.chdir(here)

    p = argparse.ArgumentParser(description="T5 新闻标题/摘要生成（默认可交互选样例）")
    p.add_argument("--config", type=str, default=None, help="配置文件路径（如 configs/default.yaml），覆盖推理默认值")
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

    # 如果指定了 --config，从 YAML 加载推理默认值（仅当命令行未指定时生效）
    if args.config:
        cfg = load_config(args.config)
        if args.max_new_tokens == PRED_MAX_NEW_TOKENS:
            args.max_new_tokens = cfg.inference.max_new_tokens
        if args.num_beams == 4:
            args.num_beams = cfg.inference.num_beams
        if args.length_penalty == PRED_LENGTH_PENALTY:
            args.length_penalty = cfg.inference.length_penalty
        if args.no_repeat_ngram == 2:
            args.no_repeat_ngram = cfg.inference.no_repeat_ngram

    ck = _ckpt_dir(args.ckpt)
    if not os.path.isdir(ck) or not os.path.isfile(os.path.join(ck, "config.json")):
        print(f"未找到合法 checkpoint: {ck}\n请先运行: python train.py", file=sys.stderr)
        sys.exit(1)

    samples_path = args.samples if os.path.isabs(args.samples) else os.path.join(here, args.samples)
    sample_items, n_sample = load_sample_articles(samples_path)

    hp = load_hparams(ck)
    prefix = (hp.get("prefix") or DEFAULT_PREFIX) if isinstance(hp, dict) else DEFAULT_PREFIX

    dev = get_device()
    print("设备:", dev, flush=True)
    tok = AutoTokenizer.from_pretrained(ck, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(ck)
    model.to(dev)
    model.eval()

    def run_one(article: str) -> str:
        s = (article or "").strip()
        if not s:
            return ""
        line = prefix + s
        enc = tok(
            line,
            max_length=hp.get("max_source_len", 512) or 512,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(dev) for k, v in enc.items()}
        ngram = args.no_repeat_ngram if int(args.no_repeat_ngram) > 0 else None
        gen_kw: Dict[str, Any] = {
            "max_new_tokens": int(args.max_new_tokens),
            "num_beams": int(args.num_beams),
            "early_stopping": True,
        }
        if ngram is not None:
            gen_kw["no_repeat_ngram_size"] = ngram
        if int(args.num_beams) > 1:
            gen_kw["length_penalty"] = float(args.length_penalty)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kw)
        return tok.decode(out[0], skip_special_tokens=True).strip()

    if args.once:
        t = (args.text or "").strip()
        if not t:
            t = (sys.stdin.readline() or "").strip()
        if not t:
            print("请提供 --text，或向 stdin 输入一行正文", file=sys.stderr)
            sys.exit(1)
        print(run_one(t), flush=True)
        return

    if n_sample > 0:
        print(
            f"已加载样例文件（共 {n_sample} 条）: {os.path.abspath(samples_path)}\n"
            f"  输入 1–{n_sample}：用其中一条「正文」做生成；\n"
            f"  输入 t：多行自输入一段正文，单独一行只按回车结束；\n"
            f"  输入 h：帮助；q：退出。"
        )
    else:
        print(
            f"未找到 {os.path.abspath(samples_path)}，请先运行 prepare_data 生成 20 条样例；\n"
            f"  仍可：输入 t 多行自输入，或 q 退出。",
        )

    while True:
        try:
            s = input("文章> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not s:
            continue
        low = s.lower()
        if low in ("q", "quit", "exit"):
            print("再见。")
            break
        if low in ("h", "help", "?"):
            if n_sample > 0:
                print(
                    f"  1–{n_sample} 使用样例；t=多行输入；h=本说明；q=退出。"
                )
            else:
                print("  t=多行输入；h=本说明；q=退出。")
            continue
        if low == "t":
            print("多行输入正文，单独一行只按回车结束：", flush=True)
            lines: List[str] = []
            while True:
                line = input()
                if line.strip() == "" and len(lines) > 0:
                    break
                lines.append(line)
            block = "\n".join(lines).strip()
            if not block:
                print("(空，已跳过)", flush=True)
                continue
            out = run_one(block)
            print("生成:\n" + out + "\n", flush=True)
            continue
        if s.isdigit() and n_sample > 0:
            idx = int(s)
            if 1 <= idx <= n_sample:
                art = (sample_items[idx - 1].get("article") or "").strip()
                ref = (sample_items[idx - 1].get("reference_summary") or "").strip()[:200]
                print(f"--- 样例 {idx}（前 200 字参考摘要，仅对照）: {ref!s}…", flush=True)
                out = run_one(art)
                print("生成:\n" + out + "\n", flush=True)
            else:
                print(f"请输入 1 到 {n_sample} 之间的数字。", file=sys.stderr)
            continue
        out = run_one(s)
        print("生成:\n" + out + "\n", flush=True)


if __name__ == "__main__":
    main()
