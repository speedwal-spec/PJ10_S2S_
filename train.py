# -*- coding: utf-8 -*-
"""
T5 新闻标题 / 摘要生成微调（本仓库为骨架，待补全处见下方）.

依赖：先运行 prepare_data.py 生成 data_manifest.json 并完成数据集缓存。

**必做 1**：补全 `build_preprocess_fn` 中 `_map_fn` 的实现（S2S 输入/标签）后删除 `NotImplementedError`。
**必做 2**：补全 `run_s2s_training` 内手写训练/验证/保存逻辑后删除 `NotImplementedError`。
本地试跑可减小本文件顶栏中的 MAX_TRAIN_SAMPLES / MAX_VAL_SAMPLES，或相应命令行参数。

提示：可自建 `DataLoader` + `AdamW` + 可选 `get_linear_schedule_with_warmup`；注意梯度累积、
`labels` 中 padding 为 -100 时与 `model(**batch).loss` 的用法；验证集上可算 `model(**batch).loss`（eval 模式）。
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    set_seed,
)

# =============================================================================
# 可调超参：本地试跑可先把子集设小
# =============================================================================
MANIFEST = "data_manifest.json"
OUTPUT_DIR = "t5-news-checkpoint"
MODEL_NAME = "google-t5/t5-small"

# 0 表示用该划分的全部样本；>0 时只取前 N 条（调试用）
MAX_TRAIN_SAMPLES = 2_000
MAX_VAL_SAMPLES = 400

EPOCHS = 1
BATCH_SIZE = 4
GRAD_ACCUM = 2
LR = 3e-4
MAX_SOURCE_LENGTH = 512
# 监督序列截断上限；越小越偏向短句/标题式
MAX_TARGET_LENGTH = 40
WARMUP_RATIO = 0.06
PREFIX = "summarize: "

# =============================================================================


def _abs_here(*parts: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def load_manifest(path: str) -> Dict[str, Any]:
    p = path if os.path.isabs(path) else _abs_here(path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"未找到 {p}，请先运行 prepare_data.py")
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


def build_preprocess_fn(
    text_col: str,
    summary_col: str,
    tokenizer: AutoTokenizer,
    prefix: str,
    max_source: int,
    max_target: int,
):
    """
    返回 batched `dataset.map` 要用的 _map_fn。

    须把 batch 中每条：正文经 prefix 与 tokenizer 编为 **encoder 输入**；
    参考摘要编为 `labels`（T5/Seq2Seq 训练目标，一般为摘要子词 id 列表）。
    `DataCollatorForSeq2Seq` 会把 `labels` 里的 padding 标成 -100 供 loss 忽略。

    提示：可 `with tokenizer.as_target_tokenizer():` 再 tokenize 摘要；新 API 亦可用 `text_target=...`。
    返回的 dict 通常含 `input_ids`、`attention_mask`、`labels`（与 tokenizer 对 batch 返回一致）。
    """

    def _map_fn(examples: Dict[str, List]) -> Dict[str, List]:
        # 1. 清理并拼接输入：为每一篇正文加上前缀 (如 "summarize: ")
        inputs = [prefix + str(text if text is not None else "") for text in examples[text_col]]
        targets = [str(summary if summary is not None else "") for summary in examples[summary_col]]

        # 2. 编码 Encoder 输入 (正文)
        model_inputs = tokenizer(
            inputs,
            max_length=max_source,
            truncation=True,
        )

        # 3. 编码 Decoder 标签 (参考摘要)
        labels = tokenizer(
            text_target=targets,
            max_length=max_target,
            truncation=True,
        )

        # 4. 将摘要的 token ids 赋值给 "labels" 键，供模型自动计算 CrossEntropyLoss
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    return _map_fn


def try_compute_rouge(
    preds: List[str], refs: List[str]
) -> Optional[Dict[str, float]]:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return None
    s = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for p, g in zip(preds, refs):
        sc = s.score(g, p)
        r1.append(sc["rouge1"].fmeasure)
        r2.append(sc["rouge2"].fmeasure)
        rl.append(sc["rougeL"].fmeasure)
    return {
        "rouge1": float(np.mean(r1)) if r1 else 0.0,
        "rouge2": float(np.mean(r2)) if r2 else 0.0,
        "rougeL": float(np.mean(rl)) if rl else 0.0,
    }


def run_s2s_training(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    train_tok: Any,
    val_tok: Any,
    data_collator: DataCollatorForSeq2Seq,
    val_ds_raw: Any,
    text_col: str,
    summary_col: str,
    args: argparse.Namespace,
    out_dir: str,
) -> None:
    """
    在此实现「手写」Seq2S 训练主流程，**不得**再使用 `Trainer` 封装整段训练（理解流程后可自写循环）。

    建议步骤（可调整，但需能保存权重供 predict/evaluate_rouge）：
    1.  `device =` cuda/cpu/mps；`model.to(device)`（若未在外部完成）；
    2.  `from torch.utils.data import DataLoader`，对 `train_tok` / `val_tok` 建 DataLoader，batch_size/ shuffle 用 args；
    3.  优化器例如 `torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)`；
    4.  可选 `get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)`，
        其中 `num_training_steps` 需按「每 epoch 的 optimizer 更新步数 × epoch」、并考虑**梯度累积**（见下）；
    5.  每个 epoch：train 模式、`model(...)` 前向，`outputs.loss` 或等价 loss；`loss.backward()`；
        若 `GRAD_ACCUM>1`：每累积满 `args.grad_accum` 次 `backward` 再 `optimizer.step()` 与 `scheduler.step()` 与 `zero_grad`；
    6.  可 `clip_grad_norm_`；验证集上 `model.eval()`、`torch.no_grad()` 算平均 val loss 便于调参/早停思路；
    7.  训练结束：`model.save_pretrained(out_dir)`、`tokenizer.save_pretrained(out_dir)`（main 在成功后也会再写一份保险）；
    8.  若使用 fp16，可用 `torch.cuda.amp.autocast` + `GradScaler`（非必须）。

    完成后删除本函数体中的 `NotImplementedError`。
    """
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    from tqdm import tqdm
    import math

    # 1. 自动设备识别
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model.to(device)

    # 2. 构建 DataLoader
    train_loader = DataLoader(
        train_tok, shuffle=True, batch_size=args.batch_size, 
        collate_fn=data_collator, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_tok, shuffle=False, batch_size=args.batch_size, 
        collate_fn=data_collator, pin_memory=(device.type == 'cuda')
    )

    # 3. 优化器进阶配置：分离权重衰减 (Weight Decay)
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    # 4. 学习率调度器 (带 Warmup)
    update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_training_steps = args.epochs * update_steps_per_epoch
    num_warmup_steps = int(total_training_steps * 0.06)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, total_training_steps)

    # 5. AMP 混合精度加速
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 6. 核心训练主循环
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum
            
            scaler.scale(loss).backward()
            total_train_loss += loss.item() * args.grad_accum
            
            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.set_postfix({'loss': f"{loss.item() * args.grad_accum:.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)

        # 7. 验证集评估
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False)
            for batch in val_bar:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(**batch)
                    total_val_loss += outputs.loss.item()
                    
        avg_val_loss = total_val_loss / max(1, len(val_loader))
        print(f"\n✅ Epoch {epoch+1} finished | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 8. 训练结束，保存模型
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    p = argparse.ArgumentParser(description="T5 新闻摘要/标题 微调（需补全训练与预处理）")
    p.add_argument("--manifest", type=str, default=MANIFEST)
    p.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--model_name", type=str, default=MODEL_NAME)
    p.add_argument("--max_train_samples", type=int, default=MAX_TRAIN_SAMPLES)
    p.add_argument("--max_val_samples", type=int, default=MAX_VAL_SAMPLES)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--max_source_len", type=int, default=MAX_SOURCE_LENGTH)
    p.add_argument("--max_target_len", type=int, default=MAX_TARGET_LENGTH)
    p.add_argument("--prefix", type=str, default=PREFIX)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_rouge_eval", action="store_true", help="训练结束后在验证上不算 ROUGE（省时间）")
    args = p.parse_args()

    manifest = load_manifest(args.manifest)
    text_col = manifest["text_column"]
    summary_col = manifest["summary_column"]
    set_seed(args.seed)
    print("设备:", "cuda" if torch.cuda.is_available() else "cpu", flush=True)
    print("正在从本地 cache 加载数据集（大缓存时可能 1～数分钟无新输出，属正常）…", flush=True)
    raw = _load_raw_dataset(manifest)
    print("数据集已载入，正在取 train/val 子集…", flush=True)
    train_ds = raw["train"]
    val_ds = raw["validation"]

    n_tr = int(args.max_train_samples) if int(args.max_train_samples) > 0 else len(train_ds)
    n_val = int(args.max_val_samples) if int(args.max_val_samples) > 0 else len(val_ds)
    n_tr = min(n_tr, len(train_ds))
    n_val = min(n_val, len(val_ds))
    train_ds = train_ds.select(range(n_tr))
    val_ds = val_ds.select(range(n_val))
    print(
        f"子集: train={n_tr}, val={n_val}（0 表示使用该划分全量，见顶栏默认值或 --max_train_samples / --max_val_samples）",
        flush=True,
    )

    print("正在加载 T5 分词器与预训练权重（首次会下载，可能较久）…", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    print("模型已就绪，开始 tokenize…", flush=True)

    preprocess = build_preprocess_fn(
        text_col, summary_col, tokenizer, args.prefix, args.max_source_len, args.max_target_len
    )

    train_tok = train_ds.map(
        preprocess,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="tokenize train",
    )
    val_tok = val_ds.map(
        preprocess,
        batched=True,
        remove_columns=val_ds.column_names,
        desc="tokenize val",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    out_dir = args.output_dir if os.path.isabs(args.output_dir) else _abs_here(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    run_s2s_training(
        model,
        tokenizer,
        train_tok,
        val_tok,
        data_collator,
        val_ds,
        text_col,
        summary_col,
        args,
        out_dir,
    )
    # 若 run_s2s_training 内已 save_pretrained，下述仍会再写一次作兜底
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "train_hparams.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "manifest": os.path.abspath(
                    args.manifest if os.path.isabs(args.manifest) else _abs_here(args.manifest)
                ),
                "prefix": args.prefix,
                "max_source_len": args.max_source_len,
                "max_target_len": args.max_target_len,
                "max_train_samples": n_tr,
                "max_val_samples": n_val,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("已保存:", os.path.abspath(out_dir), flush=True)

    if not args.no_rouge_eval and n_val > 0:
        model.eval()
        dev = next(model.parameters()).device
        batch_size = min(4, args.batch_size)
        gen_ids, ref_texts = [], []
        with torch.no_grad():
            for i in range(0, min(128, n_val), batch_size):
                sl = val_ds.select(range(i, min(i + batch_size, n_val)))
                ins = [args.prefix + str(t) for t in sl[text_col]]
                enc = tokenizer(
                    ins,
                    max_length=args.max_source_len,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                ).to(dev)
                g = model.generate(
                    **enc,
                    max_new_tokens=args.max_target_len,
                    num_beams=4,
                    length_penalty=0.85,
                    early_stopping=True,
                )
                gen_ids.extend(g.cpu().tolist())
                for s in sl[summary_col]:
                    ref_texts.append(str(s))
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        n_show = min(len(preds), len(ref_texts))
        r = try_compute_rouge(preds[:n_show], ref_texts[:n_show])
        if r:
            print("验证子集 ROUGE (约前 128 条):", {k: round(v, 4) for k, v in r.items()}, flush=True)
        else:
            print("未安装 rouge_score，跳过 ROUGE。可: pip install rouge-score", file=sys.stderr)
        for k in range(min(2, n_show)):
            print(
                f"\n[样例 {k+1}]\n参考: {ref_texts[k][:200]!s}…\n生成: {preds[k]!s}",
                flush=True,
            )


if __name__ == "__main__":
    main()
