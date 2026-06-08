#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5 新闻标题 / 摘要生成微调脚本

依赖：先运行 prepare_data.py 生成 data_manifest.json 并完成数据集缓存。

用法：
    python scripts/train.py --config src/configs/ablation/baseline.yaml
    python scripts/train.py --exp_id my_experiment --lr 0.001 --epochs 3
"""
import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from typing import Any, Dict, List
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    set_seed,
    get_linear_schedule_with_warmup,
)

try:
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ 警告: matplotlib 未安装，将跳过高清图表生成", file=sys.stderr)

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.configs.config_manager import load_config, load_full_config, merge_with_cli
from src.core.visualization import (
    try_compute_rouge,
    save_training_plot,
    plot_rouge_comparison,
    plot_performance_radar,
    plot_loss_rouge_evolution,
)
from src.core.quick_eval import quick_rouge_eval, generate_report_visuals


def _project_root() -> Path:
    """返回项目根目录（scripts/ 的父目录）"""
    return Path(__file__).resolve().parents[1]


def _abs_here(*parts: str) -> str:
    """获取相对于脚本所在目录的绝对路径"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def load_manifest(path: str) -> Dict[str, Any]:
    """加载数据清单文件"""
    p = Path(path)
    if not p.is_absolute():
        p = _project_root() / path
    if not p.is_file():
        raise FileNotFoundError(f"未找到 {p}，请先运行 prepare_data.py")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_raw_dataset(manifest: Dict[str, Any]):
    """从本地缓存加载数据集"""
    name = (manifest.get("dataset_name") or "cnn_dailymail").strip()
    cache_dir = manifest.get("cache_dir")
    if not cache_dir:
        raise FileNotFoundError(
            f"manifest 中 cache_dir 为空，请重新执行 prepare_data.py"
        )
    # 若 cache_dir 是相对路径，解析为项目根目录下的路径
    if not os.path.isabs(cache_dir):
        cache_dir = str(_project_root() / cache_dir)
    if not os.path.isdir(cache_dir):
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
    """构建预处理函数（S2S 输入/标签）"""
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
) -> Dict[str, List[float]]:
    """执行 S2S 训练循环"""
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=float(getattr(args, "weight_decay", 0.01)))

    # 4. 计算总步数与学习率调度器
    steps_per_epoch = len(train_loader)
    num_update_steps_per_epoch = (steps_per_epoch + args.grad_accum - 1) // args.grad_accum
    total_train_steps = num_update_steps_per_epoch * args.epochs
    num_warmup_steps = int(total_train_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_train_steps
    )

    # 5. 初始化 AMP 混合精度
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    print(f" [Prof 提示] 硬件设备: {device} | 显存优化 AMP: {'开启' if scaler else '关闭'}")
    print(f" 训练总轮数: {args.epochs} | 每轮数据包数: {steps_per_epoch} | 等效总更新步数: {total_train_steps}")
    
    train_losses_history = []
    val_losses_history = []
    tb_log_dir = os.path.join(getattr(args, "tensorboard_dir", "runs"), getattr(args, "exp_id", "default_run"))
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    global_step = 0
    
    # Early Stopping 初始化设置
    best_val_loss = float("inf")
    patience_counter = 0
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
        cycle_actual_steps = args.grad_accum  # 初始化，第一个周期开始时会重新计算
        for step, batch in enumerate(train_iter):
            batch = {k: v.to(device) for k, v in batch.items()}

            # ── 周期级统一除数：周期起始时预计算本周期实际累积步数 ──
            # 关键原理：在 cycle 开头的第一个 micro-batch（即 optimizer.zero_grad 之后）
            # 确定本周期内所有 batch 的「统一除数」，确保尾部各 batch 梯度权重完全相等。
            # 若每个 step 独立计算，尾部周期各 batch 的除数会逐批递减（从 1/6 到 1/1），
            # 导致越靠后的 batch 梯度越大——比原始统一缩到 75% 更糟糕。
            if step % args.grad_accum == 0:
                remaining_batches = steps_per_epoch - step
                cycle_actual_steps = min(args.grad_accum, remaining_batches)

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(**batch)
                    loss = outputs.loss
                loss = loss / cycle_actual_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(**batch)
                loss = outputs.loss / cycle_actual_steps
                loss.backward()

            total_train_loss += outputs.loss.item()

            max_norm = float(getattr(args, "max_grad_norm", 1.0))
            if (step + 1) % args.grad_accum == 0 or (step + 1) == steps_per_epoch:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm) 
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm) 
                    optimizer.step()

                scheduler.step()       
                optimizer.zero_grad()  
                global_step += 1
                tb_writer.add_scalar("Train/Loss", outputs.loss.item(), global_step)
                tb_writer.add_scalar("Train/LearningRate", scheduler.get_last_lr()[0], global_step)
            
            train_iter.set_postfix(loss=outputs.loss.item())

        avg_train_loss = total_train_loss / steps_per_epoch

        # --- 验证阶段 ---
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(**batch)
                else:
                    outputs = model(**batch)
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"\n✨ Epoch {epoch+1} 成果汇总 -> Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")
        tb_writer.add_scalar("Val/Loss", avg_val_loss, epoch + 1)
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        
        # Early Stopping 核心裁决逻辑
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"📉 Val Loss 创新低 ({best_val_loss:.4f})，正在封存当前最优权重...")
            # BART 需要 forced_bos_token_id=0，否则推理时解码退化
            if model.config.model_type == "bart" and model.config.forced_bos_token_id is None:
                model.config.forced_bos_token_id = 0
                model.generation_config.forced_bos_token_id = 0
            model.save_pretrained(final_out_dir)
            tokenizer.save_pretrained(final_out_dir)
        else:
            patience_counter += 1
            print(f"⚠️ Val Loss 未降低 (已连续 {patience_counter}/{patience_limit} 次)")
            if patience_counter >= patience_limit:
                print(f"🛑 触发 Early Stopping！为防止模型过拟合，训练在 Epoch {epoch+1} 提前终止。\n")
                break
                
    tb_writer.close()
    return {
        'train_losses': train_losses_history,
        'val_losses': val_losses_history,
    }


def save_hparams_json(out_dir: str, exp_id: str, args: argparse.Namespace) -> None:
    """将训练超参写入 train_hparams.json"""
    with open(os.path.join(out_dir, "train_hparams.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "exp_id": exp_id,
                "model_name": args.model_name,
                "prefix": args.prefix,
                "max_source_len": args.max_source_len,
                "max_target_len": args.max_target_len,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "epochs": args.epochs,
                "seed": args.seed,
                "warmup_ratio": args.warmup_ratio,
                "max_train_samples": args.max_train_samples,
                "max_val_samples": args.max_val_samples,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    """主函数：解析参数并启动训练"""
    # ✅ 确定项目根目录（不改变工作目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    p = argparse.ArgumentParser(description="T5 新闻摘要/标题 微调")
    p.add_argument("--config", type=str, default=None,
                   help="配置文件路径，如 src/configs/ablation/baseline.yaml")
    p.add_argument("--exp_id", type=str, default="debug_run", help="实验ID")
    p.add_argument("--manifest", type=str,
                   default=str(_project_root() / "data_manifest.json"),
                   help="data_manifest.json 路径")
    p.add_argument("--output_dir", type=str, default="checkpoints_ablation")
    p.add_argument("--model_name", type=str, default="google-t5/t5-small")
    p.add_argument("--max_train_samples", type=int, default=2000)
    p.add_argument("--max_val_samples", type=int, default=400)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--max_source_len", type=int, default=512)
    p.add_argument("--max_target_len", type=int, default=40)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--patience", type=int, default=3, help="Early Stopping 容忍次数")
    p.add_argument("--prefix", type=str, default="summarize: ")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_beams", type=int, default=4, help="Beam Search 宽度")
    p.add_argument("--length_penalty", type=float, default=0.85, help="Beam 长度惩罚")
    p.add_argument("--min_val_for_rouge", type=int, default=128)
    p.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    p.add_argument("--weight_decay", type=float, default=0.01, help="AdamW 权重衰减")
    p.add_argument("--no_rouge_eval", action="store_true", help="跳过 ROUGE 评测")
    p.add_argument("--with_bertscore", action="store_true",
                   help="在快速评测中启用 BERTScore 语义相似度（需 pip install bert-score）")
    p.add_argument("--skip_training", action="store_true",
                   help="跳过训练，仅下载并保存预训练模型权重")
    args = p.parse_args()

    # 如果指定了 --config，从YAML加载并合并
    if args.config:
        config = load_config(args.config)
        config = merge_with_cli(config, args)
        
        # 设置 HuggingFace 镜像（从配置读取 paths.yaml 中的 hf_endpoint）
        hf_endpoint = config.environment.hf_endpoint
        if hf_endpoint and not os.environ.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = hf_endpoint
            print(f"🌐 设置 HF_ENDPOINT={hf_endpoint}")
        
        # 将配置值同步回 args（仅当 CLI 未显式指定时覆盖）
        if '--exp-id' not in ' '.join(sys.argv) and '--exp_id' not in ' '.join(sys.argv):
            args.exp_id = config.id
        args.lr = config.training.lr
        args.batch_size = config.training.batch_size
        args.grad_accum = config.training.grad_accum
        args.epochs = config.training.epochs
        args.patience = config.early_stopping.patience
        args.max_train_samples = config.data.max_train_samples
        args.max_val_samples = config.data.max_val_samples
        args.max_source_len = config.data.max_source_length
        args.max_target_len = config.data.max_target_length
        args.prefix = config.data.prefix
        args.warmup_ratio = config.training.warmup_ratio
        args.seed = config.seed
        args.model_name = config.model_name
        args.num_beams = config.inference.num_beams
        args.length_penalty = config.inference.length_penalty
        args.min_val_for_rouge = config.data.min_val_for_rouge
        args.max_grad_norm = config.training.max_grad_norm
        args.weight_decay = config.training.weight_decay
        args.tensorboard_dir = config.logging.tensorboard_dir
        if args.output_dir == "checkpoints_ablation":
            args.output_dir = config.paths.output_dir
        print(f"📋 已加载配置: {args.config}")
        print(f"📋 实验ID: {config.id} | 描述: {config.description}")

    manifest = load_manifest(args.manifest)
    text_col = manifest["text_column"]
    summary_col = manifest["summary_column"]
    set_seed(args.seed)
    print("设备:", "cuda" if torch.cuda.is_available() else "cpu", flush=True)
    print("正在从本地 cache 加载数据集…", flush=True)
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
    print(f"子集: train={n_tr}, val={n_val}", flush=True)

    print("正在加载 T5 分词器与预训练权重…", flush=True)
    # PEGASUS 使用 SentencePiece，fast tokenizer 转换在 transformers 当前版本有 bug
    if "pegasus" in args.model_name.lower():
        from transformers.models.pegasus.tokenization_pegasus import PegasusTokenizer
        tokenizer = PegasusTokenizer.from_pretrained(args.model_name)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        except Exception:
            print("⚠️ 快速分词器加载失败，回退到慢速分词器", file=sys.stderr)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    print("模型已就绪，开始 tokenize…", flush=True)

    # ── 跳过训练模式：直接保存预训练权重 ──
    if getattr(args, 'skip_training', False):
        final_out_dir = os.path.join(args.output_dir, args.exp_id)
        os.makedirs(final_out_dir, exist_ok=True)
        # BART 需要 forced_bos_token_id
        if model.config.model_type == "bart" and model.config.forced_bos_token_id is None:
            model.config.forced_bos_token_id = 0
        model.save_pretrained(final_out_dir)
        tokenizer.save_pretrained(final_out_dir)
        save_hparams_json(final_out_dir, args.exp_id, args)
        print(f"⏩ 跳过训练，预训练权重已保存至: {os.path.abspath(final_out_dir)}")
        return

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

    out_dir = args.output_dir if os.path.isabs(args.output_dir) else str(_project_root() / args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    loss_history = run_s2s_training(
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
    save_hparams_json(final_out_dir, args.exp_id, args)

    # 快速 ROUGE 评测（可选）
    r = None
    if not getattr(args, 'no_rouge_eval', False) and n_val > 0:
        r, preds, ref_texts = quick_rouge_eval(
            model, tokenizer, val_ds, args, text_col, summary_col, n_val
        )

    # 生成实验报告图表
    generate_report_visuals(loss_history, r, args, final_out_dir)

    print("="*60)
    print(f"✅ 全链路运行结束，资产已封存: {os.path.abspath(final_out_dir)}")
    print("="*60)


if __name__ == "__main__":
    main()
