"""
从实验结果中提取ROUGE分数和Loss数据
"""
import os
import json
import re
from pathlib import Path

base_dir = Path("t5-news-checkpoint")

experiments = {
    "Baseline": "baseline",
    "Exp-1a (lr=0.0001)": "exp_lr_1e4",
    "Exp-1c (lr=0.001)": "exp_lr_1e3",
    "Exp-2a (bs=2,ga=4)": "exp_bs2_ga4",
    "Exp-2c (bs=8,ga=1)": "exp_bs8_ga1",
    "Exp-3a (len=256/30)": "exp_len_short",
    "Exp-3c (len=768/50)": "exp_len_long",
    "Exp-4a (no_accum)": "exp_no_accum",
    "Exp-5a (40 samples)": "exp_data_40",
    "Exp-5c (120 samples)": "exp_data_120",
}

print("=" * 80)
print("T5 消融实验结果汇总")
print("=" * 80)
print()

for exp_name, exp_dir in experiments.items():
    exp_path = base_dir / exp_dir
    
    # 查找最新的可视化目录
    viz_dirs = sorted([d for d in exp_path.iterdir() if d.is_dir() and d.name.startswith("visualization_")])
    
    if not viz_dirs:
        print(f"{exp_name}: 未找到可视化目录")
        continue
    
    latest_viz = viz_dirs[-1]
    
    # 读取 train_hparams.json 获取配置
    hparams_file = exp_path / "train_hparams.json"
    if hparams_file.exists():
        with open(hparams_file, 'r', encoding='utf-8') as f:
            hparams = json.load(f)
    
    print(f"【{exp_name}】")
    print(f"  目录: {exp_dir}")
    print(f"  可视化: {latest_viz.name}")
    
    # 尝试从图表文件名推断ROUGE（实际应该从训练日志中获取）
    # 这里我们列出该目录下的文件
    files = [f.name for f in latest_viz.iterdir()]
    print(f"  生成图表: {', '.join(files)}")
    print()

print("=" * 80)
print("提示: ROUGE分数需要从训练输出日志中手动提取")
print("=" * 80)
