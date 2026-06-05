"""
核心模块：训练内联可视化工具
职责：提供训练过程中的 Loss 曲线、ROUGE 柱状图、雷达图等快速渲染能力
不负责离线消融实验全景分析（见 generate_comparison_plots.py）
"""
import os
import sys
import numpy as np
from typing import Any, Dict, List, Optional

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ 警告: matplotlib 未安装，将跳过高清图表生成。安装命令: pip install matplotlib", file=sys.stderr)


def try_compute_rouge(preds: List[str], refs: List[str]) -> Optional[Dict[str, float]]:
    """计算 ROUGE-1/2/L 均值（委托至 metrics 模块，保持向后兼容）"""
    from src.core.metrics import compute_rouge
    result = compute_rouge(preds, refs)
    return result if result and "rouge1" in result else None


def save_training_plot(
    train_losses: List[float], val_losses: List[float],
    rouge_scores: Optional[Dict[str, float]],
    args: Any, out_dir: str,
) -> None:
    """训练 Loss 曲线 + 超参标注"""
    if not HAS_MATPLOTLIB:
        return
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
        ax.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=6)
        ax.set_title('T5 News Summarization - Training Progress', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(labelsize=10)

        param_text = (
            f"Hyperparameters:\nExp ID: {getattr(args, 'exp_id', 'default')} | Model: {args.model_name}\n"
            f"LR: {args.lr} | Batch Size: {args.batch_size} | Grad Accum: {args.grad_accum}\n"
            f"Epochs: {getattr(args, 'epochs', 'N/A')} | Source Len: {getattr(args, 'max_source_len', 'N/A')}\n"
        )
        if rouge_scores:
            rouge_text = (
                f"\nROUGE Scores (Quick Eval on val subset):\nROUGE-1: {rouge_scores.get('rouge1', 0):.4f} | "
                f"ROUGE-2: {rouge_scores.get('rouge2', 0):.4f} | ROUGE-L: {rouge_scores.get('rougeL', 0):.4f}"
            )
            param_text += rouge_text

        plt.figtext(
            0.5, 0.01, param_text, ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', edgecolor='gray', alpha=0.9),
            family='monospace'
        )
        plt.tight_layout(rect=[0, 0.15, 1, 0.95])
        plot_path = os.path.join(out_dir, "training_curve.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"可视化失败: {e}", file=sys.stderr)


def plot_rouge_comparison(rouge_scores: Dict[str, float], save_path: str = "rouge_comparison.png") -> None:
    """ROUGE 三指标柱状图"""
    if not HAS_MATPLOTLIB:
        return
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        keys = ['rouge1', 'rouge2', 'rougeL']
        current_scores = [rouge_scores.get(k, 0) for k in keys]
        x = np.arange(len(metrics))
        bars1 = ax.bar(x, current_scores, 0.4, label='Current Model', color='#4CAF50', alpha=0.8, edgecolor='black')

        for bar, val in zip(bars1, current_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_ylabel('ROUGE Score', fontsize=12)
        ax.set_title('ROUGE Metrics Evaluation', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, max(current_scores) * 1.2 if max(current_scores) > 0 else 1.0)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"ROUGE 图生成失败: {e}", file=sys.stderr)


def plot_performance_radar(metrics: Dict[str, float], save_path: str = "performance_radar.png") -> None:
    """性能雷达图"""
    if not HAS_MATPLOTLIB:
        return
    try:
        labels, values = list(metrics.keys()), list(metrics.values())
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles, values, 'o-', linewidth=2, color='#2196F3', markersize=8)
        ax.fill(angles, values, alpha=0.25, color='#2196F3')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title('Performance Radar', fontsize=14, fontweight='bold', pad=20)

        for angle, value in zip(angles[:-1], values[:-1]):
            ax.text(angle, value + 0.05, f'{value:.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"雷达图生成失败: {e}", file=sys.stderr)


def plot_loss_rouge_evolution(train_losses: List[float], val_losses: List[float],
                               save_path: str = "loss_rouge_evolution.png") -> None:
    """Loss 演化曲线（不含超参标注，简洁版）"""
    if not HAS_MATPLOTLIB:
        return
    try:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
        ax1.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=6)

        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12, color='black')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend(loc='upper right', fontsize=11)
        ax1.set_title('Training Progress: Loss Evolution', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"演化图生成失败: {e}", file=sys.stderr)
