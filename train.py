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
import os
import argparse
import json
import os
import sys
import numpy as np
import torch
from typing import Any, Dict, List, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

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
    def _map_fn(examples: Dict[str, List]) -> Dict[str, List]:
        # 1. 给每条新闻正文拼接上 T5 专用的任务前缀 (Text-to-Text 范式)
        inputs = [prefix + str(text) for text in examples[text_col]]
        targets = [str(summary) for summary in examples[summary_col]]

        # 2. 对输入端进行 Tokenize，此处不进行 Padding，留给 DataCollator 动态处理
        model_inputs = tokenizer(
            inputs, 
            max_length=max_source, 
            truncation=True, 
            padding=False
        )

        # 3. 使用规范的新版 text_target API 对目标端（标题）进行 Tokenize
        labels = tokenizer(
            text_target=targets, 
            max_length=max_target, 
            truncation=True, 
            padding=False
        )

        # 4. 组装成模型需要的标准输入格式
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
) -> None:
    # 1. 自动检测并配置硬件设备
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    )
    model.to(device)

    # 2. 构建 PyTorch DataLoader
    train_loader = DataLoader(
        train_tok, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator
    )
    val_loader = DataLoader(
        val_tok, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator
    )

    # 3. 初始化 AdamW 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 4. 计算总步数与学习率调度器
    steps_per_epoch = len(train_loader)
    num_update_steps_per_epoch = (steps_per_epoch + args.grad_accum - 1) // args.grad_accum
    total_train_steps = num_update_steps_per_epoch * args.epochs
    num_warmup_steps = int(total_train_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_train_steps
    )

    # 5. 初始化 AMP 混合精度
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    print(f" [Prof 提示] 硬件设备: {device} | 显存优化 AMP: {'开启' if scaler else '关闭'}")
    print(f" 训练总轮数: {args.epochs} | 每轮数据包数: {steps_per_epoch} | 等效总更新步数: {total_train_steps}")
    
    tb_log_dir = os.path.join("runs", getattr(args, "exp_id", "default_run"))
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    global_step = 0
    
    # ==========================================
    # 🌟 新增：Early Stopping 初始化设置
    # ==========================================
    best_val_loss = float("inf")
    patience_counter = 0
    # 默认容忍 3 个 epoch 验证集 Loss 不下降
    patience_limit = getattr(args, "patience", 3) 
    
    # 动态获取模型固化路径
    final_out_dir = os.path.join(getattr(args, "output_dir", "checkpoints_ablation"), getattr(args, "exp_id", "default_run"))
    os.makedirs(final_out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        # --- 训练阶段 ---
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()

        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", unit="batch")
        for step, batch in enumerate(train_iter):
            batch = {k: v.to(device) for k, v in batch.items()}

            is_last_batch = (step + 1) == steps_per_epoch
            actual_accum_steps = args.grad_accum
            if is_last_batch and steps_per_epoch % args.grad_accum != 0:
                actual_accum_steps = steps_per_epoch % args.grad_accum
            # ====================================

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                loss = loss / actual_accum_steps   # 修改除数
                scaler.scale(loss).backward()
            else:
                outputs = model(**batch)
                loss = outputs.loss / actual_accum_steps  # 修改除数
                loss.backward()

            total_train_loss += loss.item() * actual_accum_steps  # 修改乘数

            if (step + 1) % args.grad_accum == 0 or (step + 1) == steps_per_epoch:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                    optimizer.step()

                scheduler.step()       
                optimizer.zero_grad()  
                global_step += 1
                tb_writer.add_scalar("Train/Loss", loss.item() * args.grad_accum, global_step)
                tb_writer.add_scalar("Train/LearningRate", scheduler.get_last_lr()[0], global_step)
            
            train_iter.set_postfix(loss=loss.item() * args.grad_accum)

        avg_train_loss = total_train_loss / steps_per_epoch

        # --- 验证阶段 ---
        model.eval()
        total_val_loss = 0
        
        # 显式使用 no_grad 上下文，关闭梯度计算以节省内存与显存
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(**batch)
                else:
                    outputs = model(**batch)
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"\n✨ Epoch {epoch+1} 成果汇总 -> Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")
        tb_writer.add_scalar("Val/Loss", avg_val_loss, epoch + 1)
        
        # ==========================================
        # 🌟 新增：Early Stopping 核心裁决逻辑
        # ==========================================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"📉 Val Loss 创新低 ({best_val_loss:.4f})，正在封存当前最优权重...")
            # 只有在性能提升时，才覆写硬盘上的 Checkpoint
            model.save_pretrained(final_out_dir)
            tokenizer.save_pretrained(final_out_dir)
        else:
            patience_counter += 1
            print(f"⚠️ Val Loss 未降低 (已连续 {patience_counter}/{patience_limit} 次)")
            if patience_counter >= patience_limit:
                print(f"🛑 触发 Early Stopping！为防止模型过拟合，训练在 Epoch {epoch+1} 提前终止。\n")
                break # 打断最外层训练循环，不再继续跑多余的 Epoch
                
    # 训练彻底结束后（无论是否早停），关闭监控面板
    tb_writer.close()

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    p = argparse.ArgumentParser(description="T5 新闻摘要/标题 微调（需补全训练与预处理）")
    p.add_argument("--exp_id", type=str, default="debug_run", help="当前消融实验的唯一标识符")
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
    )
    final_out_dir = os.path.join(out_dir, args.exp_id)
    with open(os.path.join(final_out_dir, "train_hparams.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "exp_id": args.exp_id,  # 顺手把 exp_id 也记录进去
                "model_name": args.model_name,
                "prefix": args.prefix,
                "max_source_len": args.max_source_len,
                "max_target_len": args.max_target_len,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("✅ 全链路运行结束，资产已封存:", os.path.abspath(final_out_dir), flush=True)

    if not args.no_rouge_eval and n_val > 0:
        # === 新增：从硬盘重新加载最优权重 ===
        print("📥 正在重新加载表现最好的模型权重，以进行 ROUGE 快评...", flush=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(final_out_dir)
        dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
        model.to(dev)
        # ====================================
        
        model.eval()
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


if __name__ == "__main__":main()