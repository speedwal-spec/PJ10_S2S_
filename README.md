# T5 新闻摘要生成 - 多模型消融实验研究

> 基于系统性控制变量法的多模型（T5 / BART / PEGASUS）超参数与模型对比研究
> 从实验骨架到全自动消融流水线

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-4.46-ff6f00.svg)](https://huggingface.co/transformers/)

## ✨ 核心工程亮点

- 🚀 **全自动消融流水线 (`run_pipeline.py`)**：扫描 YAML 配置自动执行多模型、多种子控制变量实验，支持 `skip_training` 跳过已微调模型
- 🧩 **多模型支持**：T5-small（微调）、BART-large-cnn（零样本）、PEGASUS-cnn_dailymail（零样本），统一 YAML 配置切换
- ⚙️ **YAML 配置系统 (`src/configs/`)**：继承链式配置管理，支持 default.yaml + paths.yaml + hardware.yaml 分层合并
- 📊 **训练内联可视化 (`core/visualization.py`)**：自动生成 Loss 曲线、ROUGE 柱状图、性能雷达图
- 📈 **ROUGE + BERTScore 双维度评测**：支持标准 ROUGE 指标与语义相似度 BERTScore

## 📋 目录

- [项目概述](#项目概述)
- [核心发现](#核心发现)
- [快速开始](#快速开始)
- [实验设计](#实验设计)
- [结果汇总](#结果汇总)
- [文件结构](#文件结构)
- [使用指南](#使用指南)
- [引用与参考](#引用与参考)

---

## 🎯 项目概述

本项目针对 **CNN/DailyMail** 新闻标题自动生成任务，使用多种预训练模型（T5-small、BART-large-cnn、PEGASUS-cnn_dailymail）进行系统性消融实验研究。通过控制变量法设计了多组对照实验，探究学习率、Batch Size、数据量等关键超参数对模型性能的影响机制，并对比不同模型架构的摘要生成能力。

### 支持的模型

| 模型 | 参数量 | 训练方式 | 配置示例 |
|------|--------|----------|----------|
| `google-t5/t5-small` | 60M | 微调（从零） | `src/configs/ablation/baseline.yaml` |
| `facebook/bart-large-cnn` | 406M | 零样本（skip_training） | `src/configs/ablation/model_bart_base.yaml` |


### 研究目标

- 🔍 确定最优超参数配置组合
- 📊 量化各因素对 ROUGE 指标的影响程度
- 💡 建立可复现的基准线（Baseline）
- 🎯 为资源受限场景提供分级优化方案

### 技术栈

- **模型**: google-t5/t5-small、facebook/bart-large-cnn
- **数据集**: CNN/DailyMail 3.0.0
- **框架**: PyTorch + Hugging Face Transformers
- **配置**: PyYAML + dataclass 类型安全，支持配置继承（extends）
- **评估**: ROUGE-1/2/L + BERTScore
- **设备**: CPU / GPU 兼容
- **依赖**: transformers>=4.46, datasets>=3.1, sentencepiece（PEGASUS）, tiktoken

---

## 🏆 核心发现

### Top 3 洞察

1. **数据量是性能的决定性因素** ⭐⭐⭐⭐⭐
   - 40 样本 → ROUGE-L: 0.1567
   - 80 样本 → ROUGE-L: 0.2423 (**+54.6%**)
   - 120 样本 → ROUGE-L: 0.2571 (**+6.1%**)
   - **结论**: 优先扩充训练数据，效果最显著

2. **学习率需要精细平衡** ⭐⭐⭐⭐
   - lr=0.0001 → ROUGE-L: 0.2020 (↓16.6%，收敛不足)
   - lr=0.0003 → ROUGE-L: 0.2423 (基准，稳定)
   - lr=0.001 → ROUGE-L: **0.2655** (↑9.6%，最优但需防过拟合)
   - **结论**: 高学习率配合 early stopping 效果最佳

3. **基准配置的合理性得到验证** ⭐⭐⭐
   - bs=4 + ga=2 + len=512 取得最佳平衡
   - 梯度累积不仅是显存技巧，更是稳定性关键
   - 512 序列长度性价比最高

---

## 🚀 快速开始

### 环境准备

```bash
# 克隆仓库
git clone https://github.com/runtangtang/PJ10.git
cd PJ10/PJ10_S2S_

# 创建虚拟环境（推荐 Python 3.10+）
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 运行全自动消融流水线（推荐）

```bash
# 一键运行所有消融实验（训练 + 评测）
python run_pipeline.py

# 启用 BERTScore 语义相似度评测
python run_pipeline.py --with_bertscore
```

`run_pipeline.py` 会自动：
1. 扫描 `src/configs/ablation/` 下所有 YAML 配置
2. 对每个配置执行 训练 → 评测
3. 汇总统计报告至 `results/ablation_statistical_report.md`

### 手动运行单实验

```bash
# 1. 准备数据
python scripts/prepare_data.py

# 2. 训练模型（T5-small 微调）
python scripts/train.py --config src/configs/ablation/baseline.yaml

# 3. 跳过训练直接保存预训练权重（BART / PEGASUS）
python scripts/train.py --config src/configs/ablation/model_bart_base.yaml --skip_training

# 4. ROUGE + BERTScore 评测
python scripts/evaluate.py --ckpt checkpoints_ablation/model_bart_base_seed42 --with_bertscore --max_samples 8
```

**注意**: 
- 首次运行会自动下载数据集和模型权重
- 建议设置 `HF_ENDPOINT=https://hf-mirror.com` 加速下载（国内用户）
- BART / PEGASUS 使用 `skip_training` 跳过微调，直接零样本评测
- 训练后自动在输出目录生成可视化图表

---

## 🔬 实验设计

### 控制变量法

采用**单一变量原则**，每次仅改变一个超参数，保持其他条件不变：

| 实验组 | 变量 | 取值 | 固定参数 |
|--------|------|------|---------|
| Exp-1 | 学习率 | 0.0001, 0.0003, 0.001 | bs=4, ga=2, len=512 |
| Exp-2 | Batch Size | 2, 4, 8 | lr=0.0003, len=512 |
| Exp-3 | 序列长度 | 256, 512, 768 | lr=0.0003, bs=4 |
| Exp-4 | 梯度累积 | 1, 2 | lr=0.0003, bs=4 |
| Exp-5 | 数据量 | 40, 80, 120 | lr=0.0003, bs=4 |
| Exp-6 | **模型架构** | T5, BART, PEGASUS | 零样本（skip_training） |

### 基准配置 (Baseline)

| 参数 | 值 |
|------|-----|
| model | google-t5/t5-small |
| dataset | cnn_dailymail/3.0.0 |
| learning_rate | 0.0003 |
| batch_size | 4 |
| gradient_accumulation | 2 |
| max_source_length | 512 |
| max_target_length | 40 |
| epochs | 5 |
| warmup_ratio | 0.06 |
| optimizer | AdamW |
| mixed_precision | AMP |

---

## 📊 结果汇总

### 完整实验数据

| 实验 | ROUGE-1 | ROUGE-2 | ROUGE-L | Train Loss | Val Loss | 相对变化 |
|------|---------|---------|---------|------------|----------|---------|
| **Baseline** | 0.3053 | 0.1345 | **0.2423** | 1.8279 | 2.1721 | - |
| lr=0.0001 | 0.2546 | 0.0987 | 0.2020 | 2.3151 | 2.2188 | ↓16.6% |
| **lr=0.001** | **0.3304** | **0.1502** | **0.2655** | 1.0822 | 2.3509 | **↑9.6%** |
| bs=2,ga=4 | 0.2919 | 0.1151 | 0.2263 | 1.8223 | 2.1881 | ↓6.6% |
| bs=8,ga=1 | 0.2905 | 0.1163 | 0.2224 | 1.8238 | 2.1613 | ↓8.2% |
| len=256/30 | 0.2704 | 0.1094 | 0.2241 | 1.9763 | 2.4855 | ↓7.5% |
| len=768/50 | 0.3031 | 0.1243 | 0.2378 | 1.8208 | 2.0885 | ↓1.9% |
| no_accum | 0.2905 | 0.1163 | 0.2224 | 1.8238 | 2.1613 | ↓8.2% |
| 40 samples | 0.2250 | 0.0735 | 0.1567 | 1.8796 | 2.3658 | ↓35.3% |
| **120 samples** | **0.3287** | 0.1377 | **0.2571** | **1.7139** | **2.1263** | **↑6.1%** |

### 可视化图表

每个实验目录下包含 4 张可视化图表：

1. **training_curve.png** - 训练/验证 Loss 曲线
2. **rouge_comparison.png** - ROUGE 分数柱状对比
3. **performance_radar.png** - 多维度性能雷达图
4. **loss_rouge_evolution.png** - Loss-ROUGE 联合演化图

---

## 📁 文件结构

```
PJ10_S2S_/
├── run_pipeline.py              # ★ 全自动消融流水线（一键运行所有实验）
├── requirements.txt             #   Python 依赖管理
├── data_manifest.json           #   数据集元数据
├── sample_articles_20.json      #   20 条测试样例
│
├── scripts/                     # ★ 单体执行脚本
│   ├── prepare_data.py          #   数据准备（下载 CNN/DailyMail）
│   ├── train.py                 #   训练 + skip_training 保存
│   ├── evaluate.py              #   ROUGE + BERTScore 评测
│   └── demo.py                  #   Gradio Web 演示
│
├── src/                         # ★ 核心业务逻辑
│   ├── configs/                 #   YAML 配置系统
│   │   ├── config_manager.py    #     配置加载 & 继承解析
│   │   ├── default.yaml         #     默认超参数
│   │   ├── paths.yaml           #     路径配置
│   │   ├── hardware.yaml        #     硬件配置
│   │   └── ablation/            #     消融实验配置
│   │       ├── baseline.yaml    #       T5-small 基准
│   │       ├── model_bart_base.yaml    #  BART 零样本
│   │       └── model_pegasus_*.yaml   #  PEGASUS 零样本
│   ├── core/                    #   核心模块
│   │   ├── metrics.py           #     BERTScore 封装
│   │   ├── model_manager.py     #     多模型管理器
│   │   └── visualization.py     #     训练内联可视化
│   ├── api/                     #   RESTful API 框架
│   └── ui/                      #   Gradio 交互界面
│
├── checkpoints_ablation/        # 消融实验输出（.gitignore）
├── results/                     # 评测结果汇总
├── examples/                    # 生成样例
├── data_cache/                  # 数据集缓存（.gitignore）
│
└── README.md                    # 项目文档
```

---

## 📖 使用指南

### 1. 数据准备

```bash
python scripts/prepare_data.py
```

### 2. 全自动消融流水线

```bash
# 运行所有消融实验（推荐）
python run_pipeline.py --with_bertscore
```

流水线会自动扫描 `src/configs/ablation/` 下所有 YAML 配置，对每个配置执行训练→评测，最终生成统计报告。

对于配置了 `skip_training: true` 的模型（BART、PEGASUS），流水线会跳过微调步骤，直接下载预训练权重并评测。

### 3. 单体训练

```bash
# T5-small 微调
python scripts/train.py --config src/configs/ablation/baseline.yaml

# 跳过训练，保存预训练权重（BART / PEGASUS）
python scripts/train.py --config src/configs/ablation/model_bart_base.yaml --skip_training

# 或直接传参
python scripts/train.py --exp_id my_exp --model_name facebook/bart-large-cnn --max_train_samples 8 --no_rouge_eval --skip_training
```

### 4. ROUGE + BERTScore 评测

```bash
# 在验证集上评测
python scripts/evaluate.py --ckpt checkpoints_ablation/model_bart_base_seed42 --split validation --max_samples 1000 --with_bertscore
```

### 5. 配置管理

```bash
# 查看当前硬件配置
python -c "from src.configs.config_manager import load_full_config; c = load_full_config(); print(c.environment)"
```

### 6. 查看 TensorBoard

```bash
tensorboard --logdir=runs
```

---

## ⚙️ 配置系统详解

### 继承链

```
default.yaml ← paths.yaml ← hardware.yaml ← baseline.yaml ← model_*.yaml
```

每层配置可以部分覆盖上层，未设置的字段会从父配置继承。

### skip_training 机制

对于已在 CNN/Dailymail 上微调过的模型（BART-large-cnn、PEGASUS-cnn_dailymail），无需再次训练，通过 `training.skip_training: true` 跳过训练步骤：

```yaml
training:
  skip_training: true   # 跳过训练，仅下载 + 保存 + 评测
```

| 模型 | 是否需要训练 | skip_training |
|------|-------------|--------------|
| `t5-small` | ✅ 需要（从零微调） | `false` |
| `bart-large-cnn` | ❌ 已微调（零样本） | `true` |
| `pegasus-cnn_dailymail` | ❌ 已微调（零样本） | `true` |

- **调参方向**：尝试更多学习率（如 5e-4, 5e-5）、更长训练轮数、余弦退火调度
- **模型升级**：切换到 t5-base 或 t5-large
- **数据增强**：使用 CNN/DailyMail 全量数据或引入 xsum 数据集
- **评估扩展**：引入 BLEU、METEOR、BERTScore 等多维评估
- **可视化增强**：注意力权重可视化、生成结果对比表
- **流水线扩展**：添加超参搜索（Grid Search / Optuna）自动调优


