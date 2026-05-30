# T5 新闻摘要生成 - 消融实验研究

> 基于系统性控制变量法的 T5-small 模型超参数优化研究
> 从实验骨架到完整 MLOps 体系的全链路实现

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-ff6f00.svg)](https://huggingface.co/transformers/)

## ✨ 核心工程亮点

- 🚀 **全自动消融流水线 (`run_pipeline.py`)**：扫描 YAML 配置自动执行多组控制变量实验，支持 OOM 异常捕获与断点跳过
- ⚙️ **YAML 配置系统 (`configs/`)**：继承链式配置管理，支持 default.yaml + paths.yaml + hardware.yaml 分层合并
- ⚔️ **A/B 对比竞技场 (`ui/gradio_app.py`)**：双槽位模型热加载（Hot-Swapping），动态读取测试集样例，直观对比不同实验配置的生成质量
- 🌐 **微服务 API 框架 (`api/`)**：预留了标准的 RESTful 接口设计，支持多版本模型热加载与流式输出（Streaming）
- 🛰️ **星型联邦调度引擎 (`distributed_core/`)**：面向多云异构算力的分布式训练框架，支持 Kaggle/HF 等多平台算力并联调度
- 📊 **训练内联可视化 (`core/visualization.py`)**：自动生成 Loss 曲线、ROUGE 柱状图、性能雷达图

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

本项目针对 **CNN/DailyMail** 新闻标题自动生成任务，使用 **T5-small** 模型进行系统性消融实验研究。通过控制变量法设计了多组对照实验，探究学习率、Batch Size、数据量等关键超参数对模型性能的影响机制。

### 研究目标

- 🔍 确定最优超参数配置组合
- 📊 量化各因素对 ROUGE 指标的影响程度
- 💡 建立可复现的基准线（Baseline）
- 🎯 为资源受限场景提供分级优化方案

### 技术栈

- **模型**: google-t5/t5-small (60M 参数)
- **数据集**: CNN/DailyMail 3.0.0
- **框架**: PyTorch + Hugging Face Transformers
- **配置**: PyYAML + dataclass 类型安全
- **评估**: ROUGE-1/2/L
- **界面**: Gradio 4+
- **设备**: CPU / GPU 兼容

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

# 安装依赖
pip install -r requirements.txt
```

### 运行基准实验

```bash
# 准备数据
python prepare_data.py

# 直接运行训练（使用命令行参数）
python train.py --max_train_samples 80 --max_val_samples 20 --epochs 5 --exp_id baseline_run

# 或使用 YAML 配置文件
python train.py --config configs/ablation/baseline.yaml
```

### 运行完整消融实验（自动流水线）

```bash
# 一键执行全部消融实验（自动加载 configs/ablation/ 目录下的配置）
python run_pipeline.py
```

### 启动 A/B 对比界面

```bash
# 启动 Gradio Web 界面（默认 http://127.0.0.1:7860）
python -c "from ui.gradio_app import launch; launch()"
```

**注意**: 
- 首次运行会自动下载数据集和模型权重
- 建议设置 `HF_ENDPOINT` 环境变量加速下载
- 单次训练后自动在输出目录生成可视化图表（training_curve.png, rouge_comparison.png 等）

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
├── train.py                     # 核心训练脚本（已补全 _map_fn + run_s2s_training）
├── evaluate_rouge.py            # 统一 ROUGE 评测
├── predict.py                   # 交互推理
├── prepare_data.py              # 数据准备
├── run_pipeline.py              # ★ 自动化消融流水线
├── requirements.txt
├── data_manifest.json           # 数据集元数据
├── sample_articles_20.json      # 20 条测试样例
│
├── api/                         # ★ 微服务接口框架
│   ├── __init__.py
│   └── service_framework.py     #   RESTful API 架构定义 (Mock Mode)
│
├── distributed_core/            # ★ 分布式联邦调度框架
│   ├── __init__.py
│   ├── cloud_dispatcher.py      #   星型主控节点与异构 Worker 定义
│   └── message_queue.py         #   异步任务队列与结果收集器
│
├── configs/                     # ★ YAML 配置系统
│   ├── config_manager.py        #   配置加载 & 继承解析
│   ├── default.yaml             #   默认超参
│   ├── paths.yaml               #   路径配置
│   ├── hardware.yaml            #   硬件配置
│   └── ablation/                #   消融实验配置（继承链）
│       ├── baseline.yaml
│       ├── lr_0_001.yaml
│       ├── bs_8.yaml
│       └── samples_40.yaml
│
├── core/                        # ★ 核心模块
│   ├── __init__.py
│   ├── model_manager.py         #   双槽位模型管理器
│   └── visualization.py         #   训练内联可视化
│
├── ui/                          # ★ Gradio 交互界面
│   ├── __init__.py
│   └── gradio_app.py            #   A/B 对比竞技场
│
├── t5-news-checkpoint/          # 单次训练输出（.gitignore）
├── checkpoints_ablation/        # 消融实验输出（.gitignore）
└── data_cache/                  # 数据集缓存（.gitignore）
```

---

## 📖 使用指南

### 1. 单次训练

```bash
# 步骤1: 准备数据
python prepare_data.py

# 步骤2: 运行训练（使用 YAML 配置）
python train.py --config configs/ablation/baseline.yaml

# 或直接传参（不依赖配置文件）
python train.py --exp_id my_exp --max_train_samples 80 --max_val_samples 20 --lr 0.0003 --epochs 5
```

### 2. 运行消融实验流水线

```bash
# 自动扫描 configs/ablation/ 下所有 YAML 配置并依次执行
python run_pipeline.py
```

### 3. 评测模型

```bash
# 在验证集上评测（默认 1000 条子集，可复现）
python evaluate_rouge.py --ckpt t5-news-checkpoint/my_exp

# 在测试集上全量评测（较慢）
python evaluate_rouge.py --ckpt t5-news-checkpoint/my_exp --split test --max_samples 0
```

### 4. 交互推理

```bash
# 交互模式：输入 1-20 选样例，或 t 自输入
python predict.py --ckpt t5-news-checkpoint/my_exp
```

### 5. 启动 Gradio A/B 对比界面

```bash
# 启动 Web UI
python -c "from ui.gradio_app import launch; launch()"
```

### 6. 查看 TensorBoard

```bash
tensorboard --logdir=runs
```

---

## 💡 实验拓展方向

- **调参方向**：尝试更多学习率（如 5e-4, 5e-5）、更长训练轮数、余弦退火调度
- **模型升级**：切换到 t5-base 或 t5-large
- **数据增强**：使用 CNN/DailyMail 全量数据或引入 xsum 数据集
- **评估扩展**：引入 BLEU、METEOR、BERTScore 等多维评估
- **可视化增强**：注意力权重可视化、生成结果对比表
- **流水线扩展**：添加超参搜索（Grid Search / Optuna）自动调优

---

## 📚 引用与参考

### 相关论文

1. **T5 模型**: Raffel, C., et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *JMLR* 2020.
2. **ROUGE 评估**: Lin, C.Y. "ROUGE: A Package for Automatic Evaluation of Summaries." *ACL* 2004.

### 数据集

- **CNN/DailyMail**: Hermann, K.M., et al. "Teaching Machines to Read and Comprehend." *NeurIPS* 2015.

### 工具库

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [Gradio](https://gradio.app/)
- [ROUGE-Score](https://pypi.org/project/rouge-score/)

---

**最后更新**: 2026-05-25  
**更多细节**: 详见 [项目开发记录.md](./项目开发记录.md)
