# T5 新闻摘要 - 多模型消融实验

针对 CNN/DailyMail 新闻摘要任务，使用 T5-small、BART-large-cnn 两种模型进行控制变量消融实验，探究超参数对 ROUGE 指标的影响。

## 项目概览

| 模型 | 参数量 | 训练方式 |
|------|--------|----------|
| google-t5/t5-small | 60M | 微调 |
| facebook/bart-large-cnn | 406M | 零样本（skip_training） |

**技术栈**: PyTorch + Transformers / datasets / PyYAML / ROUGE + BERTScore

---

## 关键发现

| 因素 | 变化 | ROUGE-L 变化 | 结论 |
|------|------|-------------|------|
| 数据量 40→120 | +200% | 0.1567 → 0.2571 (+64%) | 影响最大，优先扩充数据 |
| 学习率 0.0003→0.001 | +233% | 0.2423 → 0.2655 (+9.6%) | 高学习率配合 early stopping 最佳 |
| 序列长度 256→512 | +100% | 0.2241 → 0.2423 (+8.1%) | 512 性价比最高 |
| bs=4→8 (ga=2→1) | +100% | 0.2423 → 0.2224 (-8.2%) | 梯度累积有益于稳定性 |

---

## 快速运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据
python scripts/prepare_data.py

# 3. 全自动消融（推荐）
python scripts/run_pipeline.py --with_bertscore

# 4. 手动单步
python scripts/train.py --config src/configs/ablation/baseline.yaml
python scripts/evaluate.py --ckpt checkpoints_ablation/baseline --with_bertscore

# 5. Gradio 演示
python scripts/demo.py
```

> 国内用户：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
> 设置 `HF_ENDPOINT=https://hf-mirror.com` 加速模型下载

---

## 实验设计

| 实验组 | 变量 | 取值 | 固定参数 |
|--------|------|------|---------|


---

## 结果

| 实验 | ROUGE-1 | ROUGE-2 | ROUGE-L | Train Loss | Val Loss | 相对变化 |
|------|---------|---------|---------|------------|----------|---------|
| **Baseline** | 0.3053 | 0.1345 | **0.2423** | 1.8279 | 2.1721 | - |

每个实验输出目录包含 4 张图表：training_curve / rouge_comparison / performance_radar / loss_rouge_evolution。

---

## 文件结构

```
PJ10_S2S_/
├── scripts/                     # 可执行脚本
│   ├── run_pipeline.py          #   全自动消融流水线
│   ├── prepare_data.py          #   数据准备
│   ├── train.py                 #   训练 / skip_training 保存
│   ├── evaluate.py              #   ROUGE + BERTScore 评测
│   └── demo.py                  #   Gradio 演示
├── src/
│   ├── configs/                 #   YAML 配置系统
│   │   ├── config_manager.py    #     配置加载（继承链）
│   │   ├── default.yaml         #     默认参数
│   │   ├── paths.yaml           #     路径配置
│   │   ├── hardware.yaml        #     硬件检测
│   │   └── ablation/            #     消融实验配置
│   │       ├── baseline.yaml           # T5-small 基准
│   │       ├── model_bart_base.yaml     # BART 零样本
│   │       └── model_pegasus_*.yaml     # PEGASUS 零样本
│   └── core/
│       ├── metrics.py           #   BERTScore 封装
│       ├── model_manager.py     #   多模型管理器
│       └── visualization.py     #   训练内联可视化
├── checkpoints_ablation/        # 模型输出（.gitignore）
├── results/                     # 评测结果 + 统计报告
├── examples/                    # 生成样例
├── data_cache/                  # 数据集缓存（.gitignore）
├── data_manifest.json           # 数据集元数据
├── sample_articles_20.json      # 20 条测试样例
├── requirements.txt
└── README.md
```

---

## 配置系统

配置文件继承链：
```
default.yaml ← paths.yaml ← hardware.yaml ← baseline.yaml ← model_*.yaml
```

`skip_training: true` 适用于已在 CNN/Dailymail 上微调过的模型（BART、PEGASUS），跳过训练直接评测。
