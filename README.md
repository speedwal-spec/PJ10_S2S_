# PJ10: 新闻标题自动生成 —— 基于 Seq2Seq / T5 的文本摘要系统


针对 CNN/DailyMail 新闻摘要任务，使用 T5-small模型进行控制变量消融实验，探究超参数对 ROUGE 指标的影响。



**技术栈**: PyTorch + Transformers / datasets / PyYAML / ROUGE + BERTScore

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

> 设置 `HF_ENDPOINT=https://hf-mirror.com` 加速模型下载

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

`skip_training: true` 适用于已在 CNN/Dailymail 上微调过的模型（BART），跳过训练直接评测。

---
## 结果产出

每个实验输出目录包含 4 张图表：training_curve / rouge_comparison / performance_radar / loss_rouge_evolution

运行评测后results目录下会有具体的评测信息文件，examples目录下会生成详细评测报告

运行run_pipeline.py会自动生成所有实验结果，并生成最终报告，存放在results目录下
---
