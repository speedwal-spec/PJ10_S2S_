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

## 🚀 运行指南

以下步骤从零开始，逐步完成环境配置、数据准备、训练和评测。

---

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/runtangtang/PJ10.git
cd PJ10/PJ10_S2S_

# 创建虚拟环境（推荐 Python 3.10+）
python -m venv .venv

# Windows 激活
.venv\Scripts\activate
# Linux/Mac 激活
# source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

> **国内用户加速**:
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```
>
> **HuggingFace 模型下载加速**:
> ```bash
> set HF_ENDPOINT=https://hf-mirror.com     # Windows
> # export HF_ENDPOINT=https://hf-mirror.com # Linux/Mac
> ```

---

### 2. 数据准备

```bash
python scripts/prepare_data.py
```

首次运行会自动从 HuggingFace 下载 **CNN/DailyMail 3.0.0** 数据集并缓存到 `data_cache/` 目录，同时生成以下文件：

| 生成文件 | 位置 | 说明 |
|---------|------|------|
| `data_manifest.json` | 项目根目录 | 数据集元数据（包含缓存路径、列名、切分大小） |
| `sample_articles_20.json` | 项目根目录 | 从测试集固定抽样 20 条，供快速试跑 |
| `data_cache/cnn_dailymail/` | 项目根目录 | 数据集本地缓存（约 700MB） |

**预期输出**:
```
正在下载数据集: cnn_dailymail -> ./data_cache
...
下载与元数据写入完成。
  train=287113, validation=13368, test=11490
  正文列: article, 摘要列: highlights
  已保存: /path/to/PJ10_S2S_/data_manifest.json
  测试样例: 20 条 -> /path/to/PJ10_S2S_/sample_articles_20.json
```

> **注意**: 若之前已下载过数据集，脚本会自动复用缓存，几秒内完成。

---

### 3. 全自动消融流水线（推荐）

一键运行所有消融实验（扫描配置 → 训练/保存 → 评测 → 生成统计报告）：

```bash
# 基本运行（T5 微调 + 零样本评测）
python run_pipeline.py

# 启用 BERTScore 语义相似度（推荐）
python run_pipeline.py --with_bertscore
```

#### 流水线工作流程

1. **扫描配置**: 读取 `src/configs/ablation/` 下所有 YAML 配置
2. **遍历配置**: 对每个配置，按实验设计执行一次运行
3. **训练或跳过**:
   - T5-small (`baseline.yaml`) → 执行完整微调
   - BART (`model_bart_base.yaml`) → 检测 `skip_training: true`，跳过训练，直接下载预训练权重并保存
   - PEGASUS (`model_pegasus_cnn_dailymail.yaml`) → 同上
4. **评测**: 自动调用 `evaluate.py` 计算 ROUGE-1/2/L 和 BERTScore
5. **汇总报告**: 生成 `results/ablation_statistical_report.md`

**预期输出**:
```
[1/2] 实验: model_bart_base (model_bart_base.yaml)
    种子 42
  ✅ [model_bart_base] seed=42: ROUGE-1=0.43xx, ROUGE-2=0.20xx, ROUGE-L=0.40xx

[2/2] 实验: model_pegasus_cnn_dailymail (model_pegasus_cnn_dailymail.yaml)
    种子 42
  ✅ [model_pegasus_cnn_dailymail] seed=42: ROUGE-1=0.41xx, ROUGE-2=0.19xx, ROUGE-L=0.38xx

🎉 多种子消融流水线执行完毕！
    总耗时: 15.32 分钟
    成功率: 2/2
    统计报告: E:/.../results/ablation_statistical_report.md
```

---

### 4. 手动单步运行（进阶）

如果只想运行特定模型或需要精细控制参数，可以分步执行。

#### 4.1 T5-small 微调

```bash
python scripts/train.py --config src/configs/ablation/baseline.yaml
```

此命令会：
1. 从配置文件读取超参数（lr=0.0003, bs=4, ga=2, epochs=5）
2. 从 `data_manifest.json` 加载数据集
3. 用训练集 2000 条样本、验证集 400 条执行 5 轮微调
4. 输出到 `checkpoints_ablation/baseline/`
5. 自动生成 Loss 曲线、ROUGE 柱状图等可视化图表

**关键参数说明**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | (无) | 配置文件路径，推荐使用 YAML 管理参数 |
| `--exp_id` | `debug_run` | 实验标识，决定输出目录名 |
| `--max_train_samples` | 2000 | 训练样本数（设为 0 使用全量） |
| `--max_val_samples` | 400 | 验证样本数 |
| `--epochs` | 5 | 训练轮数 |
| `--lr` | 0.0003 | 学习率 |
| `--batch_size` | 4 | 批次大小 |
| `--grad_accum` | 2 | 梯度累积步数 |
| `--no_rouge_eval` | (关闭) | 跳过训练中的快速 ROUGE 评测 |
| `--skip_training` | (关闭) | 跳过训练，仅保存预训练权重 |

#### 4.2 BART/PEGASUS 零样本（跳过训练）

对于已在 CNN/Dailymail 上微调过的模型，直接保存预训练权重，无需训练：

```bash
# BART-large-cnn
python scripts/train.py --config src/configs/ablation/model_bart_base.yaml --skip_training

# PEGASUS-cnn_dailymail
python scripts/train.py --config src/configs/ablation/model_pegasus_cnn_dailymail.yaml --skip_training
```

执行后，预训练权重会保存到 `checkpoints_ablation/model_bart_base_seed42/` 或对应 ID 的目录，供下一步评测使用。

#### 4.3 不依赖配置文件直接运行

```bash
python scripts/train.py \
    --exp_id my_custom_exp \
    --model_name google-t5/t5-small \
    --lr 0.001 \
    --batch_size 8 \
    --epochs 3 \
    --max_train_samples 80 \
    --max_val_samples 20 \
    --no_rouge_eval
```

> **注意**: 通过 `--config` 加载 YAML 后，命令行参数会自动覆盖配置文件中的值。

---

### 5. 评测模型

对已保存的 checkpoint 计算 ROUGE-1/2/L 和 BERTScore：

```bash
# 基本 ROUGE 评测
python scripts/evaluate.py --ckpt checkpoints_ablation/baseline

# 完整评测（ROUGE + BERTScore），指定评测条数
python scripts/evaluate.py \
    --ckpt checkpoints_ablation/model_bart_base_seed42 \
    --split validation \
    --max_samples 1000 \
    --seed 42 \
    --with_bertscore
```

**主要参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ckpt` | `checkpoints_ablation/baseline` | checkpoint 目录（包含 config.json） |
| `--manifest` | `data_manifest.json` | 数据集清单（默认从项目根目录读取） |
| `--split` | `validation` | 评测集，可选 `validation` / `test` |
| `--max_samples` | 1000 | 评测条数（0=全量） |
| `--batch_size` | 4 | 生成批次大小 |
| `--with_bertscore` | (关闭) | 启用 BERTScore 语义相似度 |
| `--with_llm_judge` | (关闭) | 启用 LLM 多维评分（需配置 API 密钥） |

**预期输出**:
```
📊 ROUGE 评测结果
============================================================
  ROUGE-1: 0.4321
  ROUGE-2: 0.2015
  ROUGE-L: 0.4018
  评测样本数: 1000
============================================================

📊 额外指标评测
============================================================
  BERTScore F1: 0.8654
============================================================
```

评测完成后，会自动在项目根目录生成三种报告格式：

| 报告文件 | 格式 | 说明 |
|---------|------|------|
| `results/{exp_id}_n{n}_rouge.json` | JSON | 结构化结果，可供程序读取 |
| `examples/{exp_id}_examples.md` | Markdown | 生成样例展示（符合评测要求） |
| `results/{exp_id}_n{n}_report.txt` | 文本 | 完整评测报告 |

---

### 6. 查看训练曲线

```bash
tensorboard --logdir=runs
```

打开浏览器访问 `http://localhost:6006`，可实时查看 Loss 下降曲线和验证集指标。

> 每个实验的训练日志位于 `runs/{exp_id}/` 目录下。

---

### 7. 启动 Web 演示（可选）

```bash
# 启动 Gradio Web UI
python scripts/demo.py

# 自定义端口 + 创建公开分享链接
python scripts/demo.py --port 8080 --share
```

访问 `http://127.0.0.1:7860`（默认）查看 A/B 对比竞技场，可加载不同 checkpoint 对比生成效果。

---

### 8. 配置管理与校验

```bash
# 查看当前完整配置（含硬件自动检测）
python -c "from src.configs.config_manager import load_full_config; c = load_full_config(); print(c)"

# 查看单个 YAML 文件的解析结果
python -c "from src.configs.config_manager import load_config; c = load_config('src/configs/ablation/baseline.yaml'); print(f'lr={c.training.lr}, bs={c.training.batch_size}')"`

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


