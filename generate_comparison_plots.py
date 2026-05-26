"""
综合对比可视化 - 生成所有实验的对比图表
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 实验数据
experiments = {
    'Baseline': {'rouge1': 0.3053, 'rouge2': 0.1345, 'rougeL': 0.2423, 'train_loss': 1.8279, 'val_loss': 2.1721},
    'lr=0.0001': {'rouge1': 0.2546, 'rouge2': 0.0987, 'rougeL': 0.2020, 'train_loss': 2.3151, 'val_loss': 2.2188},
    'lr=0.001': {'rouge1': 0.3304, 'rouge2': 0.1502, 'rougeL': 0.2655, 'train_loss': 1.0822, 'val_loss': 2.3509},
    'bs=2,ga=4': {'rouge1': 0.2919, 'rouge2': 0.1151, 'rougeL': 0.2263, 'train_loss': 1.8223, 'val_loss': 2.1881},
    'bs=8,ga=1': {'rouge1': 0.2905, 'rouge2': 0.1163, 'rougeL': 0.2224, 'train_loss': 1.8238, 'val_loss': 2.1613},
    'len=256/30': {'rouge1': 0.2704, 'rouge2': 0.1094, 'rougeL': 0.2241, 'train_loss': 1.9763, 'val_loss': 2.4855},
    'len=768/50': {'rouge1': 0.3031, 'rouge2': 0.1243, 'rougeL': 0.2378, 'train_loss': 1.8208, 'val_loss': 2.0885},
    'no_accum': {'rouge1': 0.2905, 'rouge2': 0.1163, 'rougeL': 0.2224, 'train_loss': 1.8238, 'val_loss': 2.1613},
    '40 samples': {'rouge1': 0.2250, 'rouge2': 0.0735, 'rougeL': 0.1567, 'train_loss': 1.8796, 'val_loss': 2.3658},
    '120 samples': {'rouge1': 0.3287, 'rouge2': 0.1377, 'rougeL': 0.2571, 'train_loss': 1.7139, 'val_loss': 2.1263},
}

output_dir = Path("t5-news-checkpoint/comparison_plots")
output_dir.mkdir(parents=True, exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def plot_rouge_comparison():
    """ROUGE分数综合对比"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    names = list(experiments.keys())
    rouge1_scores = [v['rouge1'] for v in experiments.values()]
    rouge2_scores = [v['rouge2'] for v in experiments.values()]
    rougeL_scores = [v['rougeL'] for v in experiments.values()]
    
    x = np.arange(len(names))
    width = 0.25
    
    bars1 = ax.bar(x - width, rouge1_scores, width, label='ROUGE-1', color='#4CAF50', alpha=0.8)
    bars2 = ax.bar(x, rouge2_scores, width, label='ROUGE-2', color='#2196F3', alpha=0.8)
    bars3 = ax.bar(x + width, rougeL_scores, width, label='ROUGE-L', color='#FF9800', alpha=0.8)
    
    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('ROUGE Metrics Comparison Across All Experiments', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rouge_comparison_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'rouge_comparison_all.png'}")

def plot_loss_comparison():
    """Loss对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    names = list(experiments.keys())
    train_losses = [v['train_loss'] for v in experiments.values()]
    val_losses = [v['val_loss'] for v in experiments.values()]
    
    # Train Loss
    colors_train = ['#e74c3c' if l > 2.0 else '#f39c12' if l > 1.8 else '#27ae60' for l in train_losses]
    bars1 = ax1.barh(names, train_losses, color=colors_train, alpha=0.8)
    ax1.set_xlabel('Train Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Final Train Loss by Experiment', fontsize=14, fontweight='bold')
    ax1.axvline(x=1.8279, color='red', linestyle='--', linewidth=2, label='Baseline (1.83)')
    ax1.legend(fontsize=10)
    ax1.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Val Loss
    colors_val = ['#e74c3c' if l > 2.3 else '#f39c12' if l > 2.15 else '#27ae60' for l in val_losses]
    bars2 = ax2.barh(names, val_losses, color=colors_val, alpha=0.8)
    ax2.set_xlabel('Val Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Final Val Loss by Experiment', fontsize=14, fontweight='bold')
    ax2.axvline(x=2.1721, color='red', linestyle='--', linewidth=2, label='Baseline (2.17)')
    ax2.legend(fontsize=10)
    ax2.grid(axis='x', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_comparison_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'loss_comparison_all.png'}")

def plot_performance_ranking():
    """性能排名图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 按ROUGE-L排序
    sorted_exps = sorted(experiments.items(), key=lambda x: x[1]['rougeL'], reverse=True)
    names = [e[0] for e in sorted_exps]
    scores = [e[1]['rougeL'] for e in sorted_exps]
    
    # 颜色编码
    colors = []
    for score in scores:
        if score >= 0.25:
            colors.append('#27ae60')  # 绿色 - 优秀
        elif score >= 0.23:
            colors.append('#f39c12')  # 橙色 - 良好
        else:
            colors.append('#e74c3c')  # 红色 - 需改进
    
    bars = ax.barh(names, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for i, (name, score) in enumerate(zip(names, scores)):
        change = ((score - 0.2423) / 0.2423) * 100
        label = f'{score:.4f} ({change:+.1f}%)'
        ax.text(score + 0.005, i, label, va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('ROUGE-L Score', fontsize=12, fontweight='bold')
    ax.set_title('Experiment Ranking by ROUGE-L Performance', fontsize=14, fontweight='bold')
    ax.axvline(x=0.2423, color='blue', linestyle='--', linewidth=2, label='Baseline (0.2423)')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_ranking.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'performance_ranking.png'}")

def plot_data_scaling():
    """数据量Scaling曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_points = {
        '40 samples': 0.1567,
        '80 samples': 0.2423,
        '120 samples': 0.2571
    }
    
    samples = [40, 80, 120]
    scores = [data_points[f'{s} samples'] for s in samples]
    
    ax.plot(samples, scores, 'o-', linewidth=3, markersize=12, color='#2196F3', label='ROUGE-L')
    ax.fill_between(samples, scores, alpha=0.2, color='#2196F3')
    
    # 标注关键点
    for s, score in zip(samples, scores):
        ax.annotate(f'{score:.4f}', xy=(s, score), xytext=(0, 15),
                   textcoords='offset points', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Training Samples', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROUGE-L Score', fontsize=12, fontweight='bold')
    ax.set_title('Data Scaling Law: Performance vs. Data Size', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'data_scaling_law.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'data_scaling_law.png'}")

def plot_radar_chart():
    """Top 3 实验雷达图对比"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Top 3 实验
    top3 = {
        'lr=0.001': experiments['lr=0.001'],
        '120 samples': experiments['120 samples'],
        'Baseline': experiments['Baseline']
    }
    
    categories = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Train Loss⁻¹', 'Val Loss⁻¹']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = ['#e74c3c', '#f39c12', '#2196F3']
    
    for (name, metrics), color in zip(top3.items(), colors):
        values = [
            metrics['rouge1'],
            metrics['rouge2'],
            metrics['rougeL'],
            1.0 / metrics['train_loss'],  # 反转Loss（越小越好）
            1.0 / metrics['val_loss']
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label=name, color=color, markersize=8)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 0.35)
    ax.set_title('Top 3 Experiments: Multi-dimensional Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top3_radar_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'top3_radar_comparison.png'}")

if __name__ == '__main__':
    print("=" * 60)
    print("Generating Comprehensive Comparison Plots...")
    print("=" * 60)
    print()
    
    plot_rouge_comparison()
    plot_loss_comparison()
    plot_performance_ranking()
    plot_data_scaling()
    plot_radar_chart()
    
    print()
    print("=" * 60)
    print("All comparison plots saved to:")
    print(f"  {output_dir.absolute()}")
    print("=" * 60)
