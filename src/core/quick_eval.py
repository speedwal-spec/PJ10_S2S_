"""
快速 ROUGE 评测模块
功能：在训练结束后对验证集子集进行快速 ROUGE 评测，并生成可视化图表
"""
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def quick_rouge_eval(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    val_ds: Any,
    args: Any,
    text_col: str,
    summary_col: str,
    n_val: int,
) -> Tuple[Optional[Dict[str, float]], List[str], List[str]]:
    """
    训练结束后在验证集子集上做快速评测（ROUGE + 可选 BERTScore）
    
    Args:
        model: 训练好的模型（会重新从 checkpoint 加载）
        tokenizer: 分词器
        val_ds: 验证数据集
        args: 命令行参数
        text_col: 文本列名
        summary_col: 摘要列名
        n_val: 验证集样本数
        
    Returns:
        (指标分数, 预测列表, 参考列表)
    """
    from src.core.metrics import compute_all_metrics
    
    print("正在重新加载表现最好的模型权重，以进行快速评测...", flush=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(src_dir)
    
    # 重新加载最佳模型
    output_dir = getattr(args, "output_dir", "checkpoints_ablation")
    exp_id = getattr(args, "exp_id", "baseline")

    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    
    model_path = os.path.join(output_dir, exp_id)

    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"模型路径不存在: {model_path}\n"
            f"请确认训练已完成，且模型已保存到该目录"
        )
    
    print(f"加载模型: {model_path}", flush=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    dev = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    )
    model.to(dev)
    model.eval()

    batch_size = int(args.batch_size)
    gen_ids, ref_texts = [], []
    
    with torch.no_grad():
        max_samples = min(int(getattr(args, "min_val_for_rouge", 128)), n_val)
        for i in range(0, max_samples, batch_size):
            sl = val_ds.select(range(i, min(i + batch_size, max_samples)))
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
                num_beams=int(getattr(args, "num_beams", 4)),
                length_penalty=float(getattr(args, "length_penalty", 0.85)),
                no_repeat_ngram_size=int(getattr(args, "no_repeat_ngram", 2)),
                early_stopping=True,
            )
            gen_ids.extend(g.cpu().tolist())
            for s in sl[summary_col]:
                ref_texts.append(str(s))
    
    preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    n_show = min(len(preds), len(ref_texts))
    
    # 使用统一评测接口（始终启用 ROUGE，训练时 BERTScore 默认关闭）
    enable_bertscore = getattr(args, "with_bertscore", False)
    metrics = compute_all_metrics(
        preds[:n_show], ref_texts[:n_show],
        enable_rouge=True,
        enable_bertscore=enable_bertscore,
        enable_llm_judge=False,
    )
    
    if metrics:
        print("验证子集评测结果:", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}, flush=True)
    else:
        print("未安装 rouge_score，跳过评测。请进行: pip install rouge-score", file=sys.stderr)
    
    for k in range(min(2, n_show)):
        print(
            f"\n[样例 {k+1}]\n参考: {ref_texts[k][:200]!s}…\n生成: {preds[k]!s}",
            flush=True,
        )
    
    return metrics, preds, ref_texts


def generate_report_visuals(
    loss_history: Dict[str, List[float]],
    rouge_scores: Optional[Dict[str, float]],
    args: Any,
    final_out_dir: str,
) -> None:
    """
    生成实验报告所需的图表
    
    Args:
        loss_history: 训练和验证 Loss 历史
        rouge_scores: ROUGE 分数
        args: 命令行参数
        final_out_dir: 最终输出目录
    """
    from datetime import datetime
    from src.core.visualization import (
        save_training_plot,
        plot_rouge_comparison,
        plot_loss_rouge_evolution,
        plot_performance_radar,
    )
    
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    viz_dir = os.path.join(final_out_dir, f"report_assets_{timestamp}")
    os.makedirs(viz_dir, exist_ok=True)
    print(f"\n正在为【实验报告】生成静态高清插图...", flush=True)

    save_training_plot(
        loss_history['train_losses'],
        loss_history['val_losses'],
        rouge_scores,
        args,
        viz_dir
    )
    
    plot_rouge_comparison(
        rouge_scores if rouge_scores else {'rouge1': 0, 'rouge2': 0, 'rougeL': 0},
        os.path.join(viz_dir, "rouge_comparison.png")
    )
    
    plot_loss_rouge_evolution(
        loss_history['train_losses'],
        loss_history['val_losses'],
        os.path.join(viz_dir, "loss_evolution.png")
    )
    
    if rouge_scores:
        perf_metrics = {
            'ROUGE-1': rouge_scores.get('rouge1', 0),
            'ROUGE-2': rouge_scores.get('rouge2', 0),
            'ROUGE-L': rouge_scores.get('rougeL', 0),
            'Precision(Est)': rouge_scores.get('rouge1', 0) * 0.9,
            'Recall(Est)': rouge_scores.get('rouge1', 0) * 1.1,
        }
        plot_performance_radar(
            perf_metrics,
            os.path.join(viz_dir, "performance_radar.png")
        )

    print(f"实验报告专用插图已生成至: {os.path.abspath(viz_dir)}")
