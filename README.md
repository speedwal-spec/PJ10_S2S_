# T5 新闻摘要生成 - 消融实验研究

> 基于系统性控制变量法的T5-small模型超参数优化研究

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-ff6f00.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

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

本项目针对 **CNN/DailyMail** 新闻标题自动生成任务，使用 **T5-small** 模型进行系统性消融实验研究。通过控制变量法设计了10组对照实验，探究学习率、Batch Size、序列长度、数据量等关键超参数对模型性能的影响机制。

### 研究目标

- 🔍 确定最优超参数配置组合
- 📊 量化各因素对ROUGE指标的影响程度
- 💡 建立可复现的基准线（Baseline）
- 🎯 为资源受限场景提供分级优化方案

### 技术栈

- **模型**: google-t5/t5-small (60M参数)
- **数据集**: CNN/DailyMail 3.0.0
- **框架**: PyTorch + Hugging Face Transformers
- **评估**: ROUGE-1/2/L
- **设备**: CPU/GPU兼容

---

## 🏆 核心发现

### Top 3 洞察

1. **数据量是性能的决定性因素** ⭐⭐⭐⭐⭐
   - 40样本 → ROUGE-L: 0.1567
   - 80样本 → ROUGE-L: 0.2423 (**+54.6%**)
   - 120样本 → ROUGE-L: 0.2571 (**+6.1%**)
   - **结论**: 优先扩充训练数据，效果最显著

2. **学习率需要精细平衡** ⭐⭐⭐⭐
   - lr=0.0001 → ROUGE-L: 0.2020 (↓16.6%，收敛不足)
   - lr=0.0003 → ROUGE-L: 0.2423 (基准，稳定)
   - lr=0.001 → ROUGE-L: **0.2655** (↑9.6%，最优但需防过拟合)
   - **结论**: 高学习率配合early stopping效果最佳

3. **基准配置的合理性得到验证** ⭐⭐⭐
   - bs=4 + ga=2 + len=512 取得最佳平衡
   - 梯度累积不仅是显存技巧，更是稳定性关键
   - 512序列长度性价比最高

### 性能对比图

```
ROUGE-L 排名:
🥇 lr=0.001:        0.2655 ↑9.6%
🥈 120 samples:     0.2571 ↑6.1%
🥉 Baseline:        0.2423 -
4. len=768/50:      0.2378 ↓1.9%
5. bs=2,ga=4:       0.2263 ↓6.6%
...
10. 40 samples:     0.1567 ↓35.3% ❌
```

---

## 🚀 快速开始

### 环境准备

```bash
# 克隆仓库
git clone https://github.com/runtangtang/PJ10.git
cd PJ10/PJ10_S2S_

# 切换到消融实验分支
git checkout ablation-experiment

# 安装依赖
pip install -r requirements.txt
```

### 运行基准实验

```powershell
# Windows PowerShell
.\run_baseline_quick.ps1

# 或直接运行
$env:HF_ENDPOINT = "https://hf-mirror.com"
python train.py --max_train_samples 80 --max_val_samples 20 --epochs 5
```

### 运行完整消融实验

```powershell
# 执行全部10组实验（约需1-2小时）
.\run_ablation_experiments.ps1
```

**注意**: 
- 首次运行会自动下载数据集和模型权重
- 建议设置 `HF_ENDPOINT` 环境变量加速下载
- 每个实验会生成独立的可视化图表

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

```yaml
model: google-t5/t5-small
dataset: cnn_dailymail/3.0.0
learning_rate: 0.0003
batch_size: 4
gradient_accumulation: 2
max_source_length: 512
max_target_length: 40
epochs: 5
warmup_ratio: 0.06
optimizer: AdamW
mixed_precision: AMP
```

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

每个实验目录下包含4张可视化图表：

1. **training_curve.png** - 训练/验证Loss曲线
2. **rouge_comparison.png** - ROUGE分数柱状对比
3. **performance_radar.png** - 多维度性能雷达图
4. **loss_rouge_evolution.png** - Loss-ROUGE联合演化图

查看示例：
```bash
# 基准实验可视化
ls t5-news-checkpoint/baseline/visualization_*/
```

---

## 📁 文件结构

```
PJ10_S2S_/
├── 📄 核心代码
│   ├── train.py                      # 主训练脚本（支持消融实验）
│   ├── predict.py                    # 推理预测脚本
│   ├── prepare_data.py               # 数据预处理脚本
│   └── evaluate_rouge.py             # ROUGE评估工具
│
├── 🧪 实验脚本
│   ├── run_ablation_experiments.ps1  # 批量实验执行（10组）
│   ├── run_baseline_quick.ps1        # 快速基准实验
│   └── install_dependencies.ps1      # 依赖安装脚本
│
├── 📊 分析工具
│   ├── extract_results.py            # 结果提取工具
│   └── generate_comparison_plots.py  # 综合对比可视化
│
├── 📝 文档
│   ├── README.md                     # 本文件
│   ├── 实验记录表.md                 # 完整实验数据记录
│   ├── 消融实验分析报告1.md          # 深度分析报告（341行）
│   ├── 演示文稿大纲.md               # 汇报PPT框架
│   └── 实验指导.md                   # 实验操作指南
│
├── ⚙️ 配置文件
│   ├── requirements.txt              # Python依赖
│   ├── data_manifest.json            # 数据配置清单
│   └── .gitignore                    # Git忽略规则
│
└── 📦 输出目录（未纳入Git）
    ├── t5-news-checkpoint/           # 模型权重和可视化
    │   ├── baseline/                 # 基准实验结果
    │   ├── exp_lr_1e4/               # 低学习率实验
    │   ├── exp_lr_1e3/               # 高学习率实验
    │   └── ...                       # 其他8组实验
    └── data_cache/                   # 数据集缓存
```

---

## 📖 使用指南

### 1. 复现实验

```powershell
# 步骤1: 准备数据
python prepare_data.py

# 步骤2: 运行单组实验
python train.py \
    --max_train_samples 80 \
    --max_val_samples 20 \
    --epochs 5 \
    --lr 0.0003 \
    --batch_size 4 \
    --grad_accum 2 \
    --output_dir t5-news-checkpoint/my_experiment

# 步骤3: 查看结果
ls t5-news-checkpoint/my_experiment/visualization_*/
```

### 2. 自定义实验

修改 `train.py` 中的超参数或添加新的实验组：

```python
# 在 train.py 中调整参数
EPOCHS = 7                           # 训练轮数
BATCH_SIZE = 8                       # Batch Size
LR = 5e-4                            # 学习率
MAX_SOURCE_LENGTH = 768              # 输入长度
```

### 3. 生成对比图表

```bash
# 生成所有实验的综合对比图
python generate_comparison_plots.py

# 输出位置: t5-news-checkpoint/comparison_plots/
```

### 4. 导出结果

```bash
# 提取所有实验的ROUGE分数
python extract_results.py
```

---

## 💡 最佳实践建议

### 短期优化（立即实施）

```yaml
学习率: 0.001
Early Stopping: patience=3
预期提升: +6-10%
实施难度: ⭐
```

### 中期优化（需额外资源）

```yaml
数据量: 200-300样本
学习率调度: cosine decay with warmup
预期提升: +15-20%
实施难度: ⭐⭐⭐
```

### 长期方向

- 升级到 t5-base 模型（220M参数）
- 多任务联合训练（摘要+分类）
- 注意力机制可视化分析
- 引入BLEU、METEOR等多维度评估

---

## 📚 引用与参考

### 相关论文

1. **T5模型**: Raffel, C., et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *JMLR* 2020.
2. **ROUGE评估**: Lin, C.Y. "ROUGE: A Package for Automatic Evaluation of Summaries." *ACL* 2004.
3. **Scaling Law**: Kaplan, J., et al. "Scaling Laws for Neural Language Models." *arXiv* 2020.

### 数据集

- **CNN/DailyMail**: Hermann, K.M., et al. "Teaching Machines to Read and Comprehend." *NeurIPS* 2015.
- 版本: 3.0.0
- 许可证: Apache 2.0

### 工具库

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [Datasets](https://huggingface.co/docs/datasets/)
- [ROUGE-Score](https://pypi.org/project/rouge-score/)

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 报告问题

- 描述具体问题现象
- 提供复现步骤
- 附上错误日志

### 提交代码

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 👥 作者

- **runtangtang** - [GitHub Profile](https://github.com/runtangtang)

---

## 🙏 致谢

感谢以下开源项目的支持：
- Hugging Face Team - Transformers库
- Google Research - T5模型
- CNN/DailyMail数据集提供者

---

## 📞 联系方式

- GitHub Issues: [提交问题](https://github.com/runtangtang/PJ10/issues)
- Email: [你的邮箱]

---

**最后更新**: 2026-05-26  
**分支**: `ablation-experiment`  
**Commit**: `bb4539c`

---

<div align="center">

**如果这个项目对你有帮助，请考虑给它一个 ⭐ Star！**

[⬆ 回到顶部](#t5-新闻摘要生成---消融实验研究)

</div>
