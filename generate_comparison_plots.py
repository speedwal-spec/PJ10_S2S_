"""
🚀 满血版 MLOps 离线分析与可视化引擎 (Auto-Discovery Version)
特性：
1. 彻底移除硬编码，自动扫描流水线产出的 JSON 指标。
2. 自动联表查询：跨目录读取 ROUGE 结果与训练超参数（train_hparams.json）。
3. 容错渲染：无论跑完几组实验，哪怕中途有崩溃缺失，都能有多少画多少。
"""

import os
import json
import matplotlib
matplotlib.use('Agg') # 无头模式，确保在无 GUI 的服务器上也能作图
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ==========================================
# 1. 核心数据引擎 (Data Ingestion Engine)
# ==========================================
def load_dynamic_results(results_dir=".", ckpt_base="checkpoints_ablation"):
    """
    动态扫描目录下的所有实验指标，并与 Checkpoint 中的超参数进行 Join。
    """
    experiments = {}
    
    # 遍历当前目录下所有流水线产出的评测文件
    for file in os.listdir(results_dir):
        if file.startswith("results_") and file.endswith(".json"):
            exp_id = file.replace("results_", "").replace(".json", "")
            
            # 读取 ROUGE 指标
            try:
                with open(os.path.join(results_dir, file), "r", encoding="utf-8") as f:
                    rouge_data = json.load(f)
            except Exception as e:
                print(f"⚠️ 警告: 读取 {file} 失败，跳过该组 ({e})")
                continue
                
            # 读取训练时的超参数快照
            hparams = {}
            hparams_path = os.path.join(ckpt_base, exp_id, "train_hparams.json")
            if os.path.exists(hparams_path):
                try:
                    with open(hparams_path, "r", encoding="utf-8") as f:
                        hparams = json.load(f)
                except Exception:
                    pass
            
            # 组装扁平化的实验资产字典
            experiments[exp_id] = {
                'rouge1': rouge_data.get('rouge1', 0),
                'rouge2': rouge_data.get('rouge2', 0),
                'rougeL': rouge_data.get('rougeL', 0),
                'lr': hparams.get('lr', 'N/A'),
                'bs': hparams.get('batch_size', 'N/A'),
                'samples': rouge_data.get('n_samples', hparams.get('max_train_samples', 0))
            }
            
    return experiments

# ==========================================
# 2. 全局样式配置 (Aesthetics)
# ==========================================
OUTPUT_DIR = Path("comparison_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 强制使用更清晰的无衬线字体，并解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 提升全局清晰度
plt.rcParams['figure.dpi'] = 150

# ==========================================
# 3. 动态渲染模块 (Dynamic Plotters)
# ==========================================
def plot_rouge_comparison(data_dict):
    if not data_dict: return
    fig, ax = plt.subplots(figsize=(max(10, len(data_dict)*1.5), 7))
    
    names = list(data_dict.keys())
    # 为了美观，去掉名字里冗长的 "Exp-" 前缀
    clean_names = [n.replace("Exp-", "") for n in names]
    
    r1 = [v['rouge1'] for v in data_dict.values()]
    r2 = [v['rouge2'] for v in data_dict.values()]
    rL = [v['rougeL'] for v in data_dict.values()]
    
    x = np.arange(len(names))
    width = 0.25
    
    ax.bar(x - width, r1, width, label='ROUGE-1', color='#4CAF50', alpha=0.85, edgecolor='white')
    ax.bar(x,         r2, width, label='ROUGE-2', color='#2196F3', alpha=0.85, edgecolor='white')
    ax.bar(x + width, rL, width, label='ROUGE-L', color='#FF9800', alpha=0.85, edgecolor='white')
    
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Matrix: ROUGE Metrics Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(clean_names, rotation=30, ha='right', fontsize=10)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / '1_rouge_comparison_dynamic.png'
    plt.savefig(save_path)
    plt.close()
    print(f"📈 [1/4] 生成 ROUGE 柱状图 -> {save_path.name}")

def plot_performance_ranking(data_dict):
    if not data_dict: return
    fig, ax = plt.subplots(figsize=(10, max(6, len(data_dict)*0.6)))
    
    # 核心：动态按 ROUGE-L 降序排列
    sorted_exps = sorted(data_dict.items(), key=lambda x: x[1]['rougeL'], reverse=True)
    clean_names = [e[0].replace("Exp-", "") for e in sorted_exps]
    scores = [e[1]['rougeL'] for e in sorted_exps]
    
    # 获取最高分作为 Benchmark
    best_score = scores[0] if scores else 0
    
    colors = ['#27ae60' if s >= best_score*0.95 else '#f39c12' if s >= best_score*0.85 else '#e74c3c' for s in scores]
    
    bars = ax.barh(clean_names, scores, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    ax.invert_yaxis()  # 让第一名排在最上面
    
    for i, score in enumerate(scores):
        ax.text(score + 0.002, i, f'{score:.4f}', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('ROUGE-L Score', fontsize=12, fontweight='bold')
    ax.set_title('Leaderboard: Sorted by ROUGE-L', fontsize=14, fontweight='bold', pad=15)
    if best_score > 0:
        ax.axvline(x=best_score, color='blue', linestyle='--', linewidth=1.5, alpha=0.5, label='SOTA')
    
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    save_path = OUTPUT_DIR / '2_performance_leaderboard.png'
    plt.savefig(save_path)
    plt.close()
    print(f" [2/4] 生成排行榜横向图 -> {save_path.name}")

def plot_data_scaling(data_dict):
    if not data_dict: return
    # 筛选出有样本量数据的组
    valid_data = {k: v for k, v in data_dict.items() if isinstance(v['samples'], int) and v['samples'] > 0}
    if not valid_data: return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 按样本量排序
    sorted_by_samples = sorted(valid_data.items(), key=lambda x: x[1]['samples'])
    samples = [e[1]['samples'] for e in sorted_by_samples]
    scores = [e[1]['rougeL'] for e in sorted_by_samples]
    
    ax.plot(samples, scores, marker='o', linestyle='-', linewidth=3, markersize=10, color='#9b59b6')
    ax.fill_between(samples, scores, alpha=0.15, color='#9b59b6')
    
    for s, score in zip(samples, scores):
        ax.annotate(f'{score:.3f}', xy=(s, score), xytext=(0, 12),
                   textcoords='offset points', ha='center', fontweight='bold')
    
    ax.set_xlabel('Training Samples', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROUGE-L Score', fontsize=12, fontweight='bold')
    ax.set_title('Data Scaling Law (Auto-Extracted)', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / '3_data_scaling_law.png'
    plt.savefig(save_path)
    plt.close()
    print(f" [3/4] 生成数据 Scaling 曲线 -> {save_path.name}")

def export_markdown_leaderboard(data_dict):
    """根据最新跑出的 JSON，自动生成 Markdown 实验表格"""
    if not data_dict: return
    
    sorted_exps = sorted(data_dict.items(), key=lambda x: x[1]['rougeL'], reverse=True)
    
    md_content = "## 🏆 自动化消融实验指标排行榜 (Auto-Generated)\n\n"
    md_content += "| 排名 | 实验 ID | LR | Batch | 训练样本 | ROUGE-1 | ROUGE-2 | ROUGE-L |\n"
    md_content += "|---|---|---|---|---|---|---|---|\n"
    
    for i, (name, m) in enumerate(sorted_exps):
        md_content += f"| {i+1} | `{name}` | {m.get('lr','-')} | {m.get('bs','-')} | {m.get('samples','-')} | **{m['rouge1']:.4f}** | {m['rouge2']:.4f} | **{m['rougeL']:.4f}** |\n"
    
    with open("Auto_Leaderboard.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print("📝 [Bonus] 已自动生成 Markdown 排行榜 -> Auto_Leaderboard.md")
    
def plot_radar_chart(data_dict):
    if len(data_dict) < 3: return # 数据太少画雷达图没意义
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # 自动提取前 3 名
    sorted_exps = sorted(data_dict.items(), key=lambda x: x[1]['rougeL'], reverse=True)[:3]
    categories = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1] # 闭合雷达图
    
    colors = ['#e74c3c', '#f39c12', '#2196F3']
    
    for idx, (name, metrics) in enumerate(sorted_exps):
        values = [metrics['rouge1'], metrics['rouge2'], metrics['rougeL']]
        values += values[:1]
        
        clean_name = name.replace("Exp-", "")
        ax.plot(angles, values, 'o-', linewidth=2.5, label=clean_name, color=colors[idx], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    # 动态计算 Y 轴上限
    max_val = max([e[1]['rouge1'] for e in sorted_exps]) * 1.1
    ax.set_ylim(0, max_val)
    ax.set_title('Top 3 Models: Multi-Dimensional Radar', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / '4_top3_radar.png'
    plt.savefig(save_path)
    plt.close()
    print(f"🕸️  [4/4] 生成 Top3 性能雷达图 -> {save_path.name}")

if __name__ == '__main__':
    print("=" * 60)
    print(" MLOps 自动化图表渲染引擎启动...")
    print("=" * 60)
    
    # 1. 动态加载数据
    dynamic_data = load_dynamic_results()
    
    if not dynamic_data:
        print(" 未在当前目录检测到任何 'results_*.json' 文件。")
        print(" 请确保已执行 `python run_pipeline.py` 并成功产出了评估 JSON。")
    else:
        print(f" 成功提取到 {len(dynamic_data)} 组实验数据！开始渲染...")
        
        # 2. 按需渲染图表
        plot_rouge_comparison(dynamic_data)
        plot_performance_ranking(dynamic_data)
        plot_data_scaling(dynamic_data)
        plot_radar_chart(dynamic_data)
        export_markdown_leaderboard(dynamic_data)
        print("\n" + "=" * 60)
        print(f"🎉 所有图表均已根据最新 JSON 动态渲染至: {OUTPUT_DIR.absolute()}")
        print("=" * 60)