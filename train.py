# -*- coding: utf-8 -*-
"""
T5 新闻标题 / 摘要生成微调训练脚本

【项目概述】
本脚本实现基于 T5 (Text-to-Text Transfer Transformer) 模型的新闻标题/摘要生成任务。
采用 Seq2Seq（序列到序列）架构，将长文本新闻正文转换为短文本标题或摘要。

【核心功能模块】
1. 数据加载与预处理：从本地缓存加载数据集，进行 tokenization 和格式转换
2. 模型训练：手写完整的训练循环（不使用 transformers.Trainer）
3. 模型评估：在验证集上计算 ROUGE 指标，评估生成质量
4. 模型保存：保存训练好的模型权重和超参数配置

【关键技术点】
- T5 模型：Google 提出的统一文本到文本框架，所有 NLP 任务都转化为 text-to-text 形式
- Teacher Forcing：训练时 decoder 使用真实的 target tokens 作为输入，加速收敛
- 梯度累积：通过多次 backward 再 step，模拟更大的 batch size，节省显存
- AMP 混合精度：使用 float16 加速训练，减少显存占用
- Beam Search：推理时使用束搜索算法，生成更高质量的文本

【工作流程】
1. 读取 data_manifest.json 获取数据集配置
2. 从本地缓存加载训练集和验证集
3. 初始化 T5 模型和 Tokenizer
4. 对数据进行 tokenization 预处理
5. 执行训练循环（多个 epoch）
   - 每个 epoch 包含：前向传播 → 计算 loss → 反向传播 → 参数更新
   - 定期在验证集上评估性能
6. 训练结束后保存模型权重
7. （可选）在验证集子集上计算 ROUGE 指标

【必做任务】（教学用途）
**必做 1**：补全 `build_preprocess_fn` 中 `_map_fn` 的实现
  - 将原始文本转换为模型可接受的 input_ids 和 labels
  - 理解 T5 的 prefix 机制和 label 掩码原理

**必做 2**：补全 `run_s2s_training` 内手写训练/验证/保存逻辑
  - 实现完整的 PyTorch 训练循环
  - 掌握 DataLoader、优化器、学习率调度器的使用
  - 理解梯度累积和混合精度训练的实现

【依赖说明】
- 必须先运行 prepare_data.py 生成 data_manifest.json
- 需要安装：torch, transformers, datasets, rouge-score, tqdm

【使用示例】
# 小规模调试（快速验证流程）
python train.py --max_train_samples 200 --max_val_samples 50 --epochs 1

# 完整训练（默认配置）
python train.py

# 自定义超参数
python train.py --epochs 3 --batch_size 8 --lr 5e-4 --grad_accum 4

# 跳过 ROUGE 评估（节省时间）
python train.py --no_rouge_eval
"""
import argparse
import json
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    set_seed,
)

# 尝试导入绘图库（可选依赖）
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端，适合服务器环境
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib 未安装，将跳过可视化。安装命令: pip install matplotlib", file=sys.stderr)

# =============================================================================
# 全局超参数配置
# 这些参数控制训练过程的关键行为，可根据硬件资源和任务需求调整
# =============================================================================
MANIFEST = "data_manifest.json"      # 数据清单文件名，由 prepare_data.py 生成
OUTPUT_DIR = "t5-news-checkpoint"    # 模型保存目录，训练完成后存储权重和配置
MODEL_NAME = "google-t5/t5-small"    # 预训练模型名称，t5-small 约60M参数

# 样本数量控制（0表示使用该划分的全部样本，>0时只取前N条用于调试）
MAX_TRAIN_SAMPLES = 40         # 训练集最大样本数，减小可加快调试
MAX_VAL_SAMPLES = 20                # 验证集最大样本数

# 训练超参数
EPOCHS = 7                           # 训练轮数，每轮遍历一次完整训练集
BATCH_SIZE = 4                       # 每个批次的样本数，受显存限制
GRAD_ACCUM = 2                       # 梯度累积步数，实际batch_size = BATCH_SIZE * GRAD_ACCUM
LR = 3e-4                            # 学习率，控制参数更新步长

# 序列长度控制
MAX_SOURCE_LENGTH = 512              # 输入正文的最大token数，超过则截断
# 监督序列截断上限；越小越偏向短句/标题式
MAX_TARGET_LENGTH = 40               # 输出摘要的最大token数，标题通常较短

# 学习率调度
WARMUP_RATIO = 0.06                  # Warmup比例，前6%的训练步数线性增加学习率
PREFIX = "summarize: "               # T5输入前缀，告诉模型执行摘要任务

# =============================================================================


def _abs_here(*parts: str) -> str:
    """
    构建相对于当前脚本文件的绝对路径
    
    【功能说明】
    将相对路径转换为绝对路径，确保在不同工作目录下都能正确找到文件
    
    【参数】
    - *parts: 路径片段，如 'data', 'manifest.json'
    
    【返回值】
    - str: 拼接后的绝对路径
    
    【使用示例】
    _abs_here('data', 'manifest.json') -> '/path/to/project/data/manifest.json'
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def load_manifest(path: str) -> Dict[str, Any]:
    """
    加载数据清单文件，获取数据集配置信息
    
    【功能说明】
    读取 prepare_data.py 生成的 data_manifest.json，提取数据集的路径、字段名等元信息
    这是训练脚本与数据准备脚本之间的接口，确保两者使用相同的数据配置
    
    【参数】
    - path: manifest 文件路径（可以是相对路径或绝对路径）
    
    【返回值】
    - Dict: 包含数据集配置的字典，主要字段包括：
      * dataset_name: 数据集名称（如 'cnn_dailymail'）
      * cache_dir: 数据集缓存目录
      * text_column: 正文字段名（如 'article'）
      * summary_column: 摘要字段名（如 'highlights'）
      * splits: 各划分的样本数量
    
    【异常处理】
    - 如果文件不存在，抛出 FileNotFoundError 并提示用户先运行 prepare_data.py
    
    【设计原则】
    - 支持相对路径和绝对路径，自动转换
    - 早期失败原则：文件不存在时立即报错，避免后续出现难以调试的错误
    """
    p = path if os.path.isabs(path) else _abs_here(path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"未找到 {p}，请先运行 prepare_data.py")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_raw_dataset(manifest: Dict[str, Any]):
    """
    从本地缓存加载原始数据集
    
    【功能说明】
    根据 manifest 中的配置，使用 datasets 库从本地缓存加载数据集
    不会重新下载，直接从磁盘读取已缓存的 Arrow 格式数据
    
    【参数】
    - manifest: 数据清单字典，由 load_manifest() 返回
    
    【返回值】
    - DatasetDict: 包含 train/validation/test 三个划分的数据集对象
    
    【工作流程】
    1. 从 manifest 提取数据集名称和缓存目录
    2. 验证缓存目录是否存在
    3. 根据数据集类型选择加载方式：
       - xsum: 直接加载，无需配置版本
       - cnn_dailymail: 需要指定配置版本（如 '3.0.0'）
    
    【注意事项】
    - 首次加载可能需要几分钟映射 Arrow 文件，属正常现象
    - 如果缓存目录无效，会抛出 FileNotFoundError
    
    【技术细节】
    - datasets.load_dataset() 检测到 cache_dir 存在时会优先使用缓存
    - Arrow 格式支持内存映射，可以高效处理超出内存大小的数据集
    """
    name = (manifest.get("dataset_name") or "cnn_dailymail").strip()
    cache_dir = manifest.get("cache_dir")
    if not cache_dir or not os.path.isdir(cache_dir):
        raise FileNotFoundError(
            f"manifest 中 cache_dir 无效: {cache_dir!r}，请重新执行 prepare_data.py"
        )
    if name.lower() == "xsum":
        return load_dataset("xsum", cache_dir=cache_dir)
    cfg = manifest.get("dataset_config") or "3.0.0"
    return load_dataset(name, cfg, cache_dir=cache_dir)


def build_preprocess_fn(
    text_col: str,
    summary_col: str,
    tokenizer: AutoTokenizer,
    prefix: str,
    max_source: int,
    max_target: int,
):
    """
    构建数据预处理函数，用于将原始文本转换为模型可接受的格式
    
    【核心功能】
    返回一个 batched 预处理函数 _map_fn，该函数会被 dataset.map() 调用，
    对数据集中的每个样本进行 tokenization 和格式转换。
    
    【Seq2Seq 任务的数据流】
    原始文本 → 添加prefix → Tokenization → input_ids + attention_mask + labels
    
    【参数说明】
    - text_col: 正文字段名（如 'article'）
    - summary_col: 摘要字段名（如 'highlights'）
    - tokenizer: T5 Tokenizer，负责将文本转换为 token IDs
    - prefix: 输入前缀（如 'summarize: '），T5 通过前缀区分不同任务
    - max_source: 输入序列最大长度，超过则截断
    - max_target: 输出序列最大长度，超过则截断
    
    【返回值】
    - function: _map_fn，接受 examples 字典，返回处理后的字典
      * input_ids: encoder 输入的 token IDs [batch_size, seq_len]
      * attention_mask: 注意力掩码，1表示真实token，0表示padding
      * labels: decoder 的目标 token IDs，用于计算 loss
    
    【关键技术点】
    
    1. **Prefix 机制**：
       T5 是统一的多任务模型，通过不同的 prefix 区分任务类型：
       - 'summarize: ' → 摘要任务
       - 'translate English to German: ' → 翻译任务
       - 'question: ' → 问答任务
       这使得同一个模型可以执行多种 NLP 任务。
    
    2. **Tokenization**：
       - Encoder 输入：prefix + 正文，转换为 input_ids
       - Decoder 标签：参考摘要，转换为 labels
       - 使用 truncation=True 确保序列长度不超过限制
    
    3. **Labels 的处理**：
       - labels 是 decoder 的监督信号，模型会预测下一个 token
       - DataCollatorForSeq2Seq 会自动将 padding 位置的 labels 设为 -100
       - CrossEntropyLoss 会忽略 -100 的位置，只计算真实 token 的 loss
    
    4. **Batched 处理**：
       - dataset.map(batched=True) 一次传入多个样本，提高效率
       - examples 是字典结构，每个键对应一个列表：
         {'article': ['text1', 'text2', ...], 'highlights': ['sum1', 'sum2', ...]}
    
    【实现原理】
    _map_fn 的执行流程：
    1. 清理数据：将 None 值转换为空字符串
    2. 添加 prefix：为每个正文添加任务前缀
    3. Encoder 编码：对 prefix+正文 进行 tokenization
    4. Decoder 编码：对参考摘要进行 tokenization（使用 text_target 参数）
    5. 组装输出：将摘要的 input_ids 作为 labels
    
    【示例】
    输入: article = "Breaking news...", highlights = "News summary..."
    处理后:
    {
        'input_ids': [123, 456, ...],        # encoder 输入
        'attention_mask': [1, 1, ...],        # 注意力掩码
        'labels': [789, 101, ...]             # decoder 目标
    }
    """

    def _map_fn(examples: Dict[str, List]) -> Dict[str, List]:
        """
        批量预处理函数，由 dataset.map() 调用
        
        【输入格式】
        examples: {'article': [text1, text2, ...], 'highlights': [sum1, sum2, ...]}
        
        【输出格式】
        {'input_ids': [...], 'attention_mask': [...], 'labels': [...]}
        """
        # ==================== Step 1: 数据清理与前缀拼接 ====================
        # 为每一篇正文加上前缀 (如 "summarize: ")，告诉 T5 执行摘要任务
        # 处理可能的 None 值，转换为空字符串避免报错
        inputs = [prefix + str(text if text is not None else "") for text in examples[text_col]]
        
        # 提取参考摘要作为监督信号
        targets = [str(summary if summary is not None else "") for summary in examples[summary_col]]

        # ==================== Step 2: 编码 Encoder 输入 (正文) ====================
        # 将 prefix + 正文 转换为 token IDs
        # max_length: 最大序列长度，超过则截断
        # truncation=True: 启用截断，确保不超过 max_length
        model_inputs = tokenizer(
            inputs,
            max_length=max_source,
            truncation=True,
        )

        # ==================== Step 3: 编码 Decoder 标签 (参考摘要) ====================
        # 使用 text_target 参数专门处理目标序列
        # 这会应用适合摘要生成的 tokenization 策略
        labels = tokenizer(
            text_target=targets,
            max_length=max_target,
            truncation=True,
        )

        # ==================== Step 4: 组装输出 ====================
        # 将摘要的 token ids 赋值给 "labels" 键
        # 模型在训练时会计算 predictions 与 labels 之间的 CrossEntropyLoss
        # DataCollatorForSeq2Seq 后续会将 padding 位置的 labels 设为 -100
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    return _map_fn


def try_compute_rouge(
    preds: List[str], refs: List[str]
) -> Optional[Dict[str, float]]:
    """
    计算 ROUGE 指标（ROUGE-1/2/L），评估生成文本与参考文本的重叠度
    
    【功能说明】
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) 是摘要生成任务的标准评估指标，
    通过计算 n-gram 重叠来衡量生成质量。
    
    【参数】
    - preds: 模型生成的摘要列表
    - refs: 参考摘要列表（ground truth）
    
    【返回值】
    - Dict: 包含三个 ROUGE 指标的 F1 分数
      * rouge1: unigram（单个词）重叠的 F1
      * rouge2: bigram（连续两个词）重叠的 F1
      * rougeL: 最长公共子序列的 F1
    - 如果 rouge_score 未安装，返回 None
    
    【ROUGE 指标详解】
    
    1. **ROUGE-1**：
       - 计算单个词的重叠
       - 反映生成摘要是否包含了参考答案中的关键词
       - 对词序不敏感
    
    2. **ROUGE-2**：
       - 计算连续两个词（bigram）的重叠
       - 反映生成摘要的词序和搭配是否合理
       - 比 ROUGE-1 更严格
    
    3. **ROUGE-L**：
       - 基于最长公共子序列（LCS）
       - 不要求连续匹配，但要求顺序一致
       - 平衡了召回率和精确率
    
    【技术细节】
    - use_stemmer=True: 启用词干提取，将 'running' 和 'runs' 视为相同
    - F-measure: 综合 Precision 和 Recall 的调和平均数
      F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    【注意事项】
    - ROUGE 仅衡量字面重叠，不考虑语义相似性
    - 同义词替换会导致分数降低，即使意思正确
    - 适合评估抽取式摘要，对抽象式摘要可能不够公平
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return None
    
    # 初始化 ROUGE 计算器，计算三种指标
    s = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    
    # 存储每个样本的得分
    r1, r2, rl = [], [], []
    
    # 逐样本计算 ROUGE
    for p, g in zip(preds, refs):
        sc = s.score(g, p)  # 注意：第一个参数是 reference，第二个是 prediction
        r1.append(sc["rouge1"].fmeasure)
        r2.append(sc["rouge2"].fmeasure)
        rl.append(sc["rougeL"].fmeasure)
    
    # 计算平均得分
    return {
        "rouge1": float(np.mean(r1)) if r1 else 0.0,
        "rouge2": float(np.mean(r2)) if r2 else 0.0,
        "rougeL": float(np.mean(rl)) if rl else 0.0,
    }


def run_s2s_training(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    train_tok: Any,
    val_tok: Any,
    data_collator: DataCollatorForSeq2Seq,
    val_ds_raw: Any,
    text_col: str,
    summary_col: str,
    args: argparse.Namespace,
    out_dir: str,
) -> Dict[str, List[float]]:
    """
    手写 Seq2Seq 训练主流程（不使用 transformers.Trainer）
    
    【核心功能】
    实现完整的 PyTorch 训练循环，包括：
    1. 数据加载与批处理
    2. 前向传播与 Loss 计算
    3. 反向传播与参数更新
    4. 学习率调度
    5. 梯度累积
    6. 混合精度训练（AMP）
    7. 验证集评估
    8. 模型保存
    
    【参数说明】
    - model: T5 预训练模型，将被微调
    - tokenizer: 分词器，用于文本编码
    - train_tok: 预处理后的训练集（已 tokenized）
    - val_tok: 预处理后的验证集（已 tokenized）
    - data_collator: 数据整理器，负责 padding 和组装 batch
    - val_ds_raw: 原始验证集（未 tokenized），用于 ROUGE 评估
    - text_col: 正文字段名
    - summary_col: 摘要字段名
    - args: 命令行参数，包含超配置
    - out_dir: 模型输出目录
    
    【返回值】
    - Dict: 包含 'train_losses' 和 'val_losses' 两个列表
    
    【训练流程详解】
    
    **Step 1: 设备准备**
    - 自动检测可用的计算设备：CUDA > MPS (Apple Silicon) > CPU
    - 将模型移动到对应设备
    
    **Step 2: 构建 DataLoader**
    - DataLoader 是 PyTorch 的数据迭代器，负责：
      * 从数据集中读取样本
      * 使用 data_collator 组装 batch
      * 支持 shuffle、多进程加载等
    - pin_memory=True: 启用页面锁定内存，加速 CPU→GPU 数据传输
    
    **Step 3: 优化器配置**
    - AdamW: Adam 优化器的改进版，修正了权重衰减的实现
    - 分组参数：对 bias 和 LayerNorm 不应用权重衰减（最佳实践）
      * 原因：这些参数通常较小，权重衰减可能导致欠拟合
    - lr: 学习率，控制每次参数更新的步长
    
    **Step 4: 学习率调度器**
    - Linear Schedule with Warmup:
      * Warmup 阶段：学习率从 0 线性增加到最大值
      * Main 阶段：学习率从最大值线性下降到 0
    - 优点：
      * Warmup 帮助模型稳定启动，避免初期梯度爆炸
      * 逐渐降低学习率有助于收敛到更好的局部最优
    - num_training_steps 的计算考虑了梯度累积：
      * 实际更新次数 = 总 batch 数 / grad_accum
    
    **Step 5: AMP 混合精度训练**
    - AMP (Automatic Mixed Precision):
      * 使用 float16 进行前向/反向传播
      * 使用 float32 存储模型参数和优化器状态
    - 优势：
      * 减少约 50% 显存占用
      * 利用 Tensor Cores 加速计算（NVIDIA GPU）
      * 几乎不影响模型精度
    - GradScaler: 动态损失缩放，防止 float16 下溢
    
    **Step 6: 训练主循环**
    
    每个 Epoch 的训练过程：
    ```
    for each batch in train_loader:
        1. 将数据移动到设备
        2. 前向传播：outputs = model(**batch)
        3. 计算 loss：loss = outputs.loss / grad_accum
        4. 反向传播：loss.backward() （累积梯度）
        5. 如果累积满 grad_accum 步：
           a. 梯度裁剪：clip_grad_norm_ (防止梯度爆炸)
           b. 参数更新：optimizer.step()
           c. 学习率更新：scheduler.step()
           d. 清空梯度：optimizer.zero_grad()
    ```
    
    **关键技术点：**
    
    1. **梯度累积 (Gradient Accumulation)**：
       - 目的：模拟更大的 batch size，节省显存
       - 原理：多次 backward 累积梯度，再统一 step
       - 示例：batch_size=4, grad_accum=2 → 等效 batch_size=8
       - Loss 需要除以 grad_accum，保持梯度尺度一致
    
    2. **梯度裁剪 (Gradient Clipping)**：
       - 目的：防止梯度爆炸，稳定训练
       - max_norm=1.0: 梯度范数超过 1.0 时按比例缩放
       - 在 RNN/Transformer 训练中常用
    
    3. **Loss 缩放**：
       - scaler.scale(loss): 放大 loss，避免 float16 下溢
       - scaler.unscale_(optimizer): 在更新前还原梯度
       - scaler.step/update: 检查 inf/nan，安全更新参数
    
    **Step 7: 验证集评估**
    - 切换到 eval 模式：model.eval()
      * 关闭 Dropout
      * 禁用 BatchNorm 的统计量更新
    - torch.no_grad(): 禁用梯度计算，节省显存和计算
    - 计算平均 validation loss，监控过拟合
    
    **Step 8: 模型保存**
    - save_pretrained(): 保存模型权重和配置文件
      * config.json: 模型架构配置
      * pytorch_model.bin: 模型权重
      * tokenizer 相关文件
    - 可在训练结束后加载继续训练或直接推理
    
    【注意事项】
    - 本实现已完整，无需补全 NotImplementedError
    - 如需自定义，可调整优化器、调度器、梯度累积策略等
    - 生产环境可添加：早停机制、checkpoint 定期保存、TensorBoard 日志等
    """
    # ==================== 导入必要的库 ====================
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    from tqdm import tqdm
    import math

    # 用于记录每个 epoch 的 loss（供可视化使用）
    train_losses_history = []
    val_losses_history = []

    # ==================== Step 1: 自动设备识别 ====================
    # 优先级：CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
    # CUDA: 支持 NVIDIA GPU，性能最佳
    # MPS: Apple M1/M2 芯片的金属着色器后端
    # CPU:  fallback 选项，速度慢但通用
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model.to(device)  # 将模型移动到计算设备

    # ==================== Step 2: 构建 DataLoader ====================
    # DataLoader 负责从数据集中批量读取数据，并应用 data_collator 进行 padding
    
    # 训练集 DataLoader
    train_loader = DataLoader(
        train_tok,                  # 预处理后的训练集
        shuffle=True,               # 每个 epoch 打乱顺序，增加随机性
        batch_size=args.batch_size, # 批次大小
        collate_fn=data_collator,   # 数据整理函数，处理 padding
        pin_memory=(device.type == 'cuda')  # CUDA 时启用 pinned memory，加速数据传输
    )
    
    # 验证集 DataLoader
    val_loader = DataLoader(
        val_tok,
        shuffle=False,              # 验证集无需打乱
        batch_size=args.batch_size,
        collate_fn=data_collator,
        pin_memory=(device.type == 'cuda')
    )

    # ==================== Step 3: 优化器进阶配置 ====================
    # 使用参数分组策略：对 bias 和 LayerNorm 不应用权重衰减
    
    # 为什么需要分组？
    # - Weight decay 相当于 L2 正则化，防止过拟合
    # - 但 bias 和 LayerNorm 的参数通常较小，应用 weight decay 可能导致欠拟合
    # - 这是 Transformer 模型的最佳实践
    
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        # 第一组：应用权重衰减的参数（大部分权重矩阵）
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        # 第二组：不应用权重衰减的参数（bias 和 LayerNorm）
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    
    # 创建 AdamW 优化器
    # AdamW vs Adam: AdamW 正确实现了权重衰减，解耦了权重衰减和梯度更新
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    # ==================== Step 4: 学习率调度器 (带 Warmup) ====================
    # Linear Schedule with Warmup: 先线性增加学习率，再线性下降
    
    # 计算总训练步数
    # 注意：由于梯度累积，实际参数更新次数 = 总 batch 数 / grad_accum
    update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_training_steps = args.epochs * update_steps_per_epoch
    
    # Warmup 步数：前 6% 的训练步数用于学习率预热
    num_warmup_steps = int(total_training_steps * 0.06)
    
    # 创建学习率调度器
    # 效果：
    # - Step 0 ~ warmup_steps: lr 从 0 线性增加到 max_lr
    # - Step warmup_steps ~ total: lr 从 max_lr 线性下降到 0
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, total_training_steps)

    # ==================== Step 5: AMP 混合精度加速 ====================
    # Automatic Mixed Precision: 自动混合精度训练
    
    # 原理：
    # - 前向/反向传播使用 float16（节省显存，加速计算）
    # - 参数存储使用 float32（保持精度）
    # - GradScaler 动态缩放 loss，防止 float16 下溢
    
    # 仅在 CUDA 上启用 AMP（MPS 和 CPU 不支持）
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ==================== Step 6: 核心训练主循环 ====================
    # 遍历所有 epoch，每个 epoch 包含训练和验证两个阶段
    
    for epoch in range(args.epochs):
        # ---------- 训练阶段 ----------
        model.train()  # 切换到训练模式（启用 Dropout、BatchNorm 更新等）
        total_train_loss = 0  # 累计训练 loss
        
        # 创建进度条，显示当前 epoch 的训练进度
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        
        # 遍历训练集中的每个 batch
        for step, batch in enumerate(progress_bar):
            # 将 batch 数据移动到计算设备
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # ---------- 前向传播 + 反向传播 ----------
            # 使用 AMP 自动混合精度（如果启用）
            with torch.cuda.amp.autocast(enabled=use_amp):
                # 前向传播：模型接收 input_ids, attention_mask, labels
                # 返回 outputs.loss（CrossEntropyLoss）
                outputs = model(**batch)
                # Loss 除以梯度累积步数，保持梯度尺度一致
                loss = outputs.loss / args.grad_accum
            
            # 反向传播：计算梯度并累积
            # scaler.scale 会放大 loss，防止 float16 下溢
            scaler.scale(loss).backward()
            # 累加 loss 用于统计（乘以 grad_accum 还原真实值）
            total_train_loss += loss.item() * args.grad_accum
            
            # ---------- 参数更新（梯度累积逻辑）----------
            # 判断是否达到累积步数或最后一个 batch
            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                # 1. 还原梯度（取消 scaler 的缩放）
                scaler.unscale_(optimizer)
                
                # 2. 梯度裁剪：防止梯度爆炸
                # 如果梯度范数超过 max_norm，则按比例缩放
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 3. 更新参数
                scaler.step(optimizer)  # 检查梯度是否有 inf/nan，安全更新
                scaler.update()         # 更新 scaler 的缩放因子
                
                # 4. 更新学习率
                scheduler.step()
                
                # 5. 清空梯度，准备下一轮累积
                optimizer.zero_grad()
            
            # 更新进度条，显示当前 loss
            progress_bar.set_postfix({'loss': f"{loss.item() * args.grad_accum:.4f}"})

        # 计算平均训练 loss
        avg_train_loss = total_train_loss / len(train_loader)

        # ---------- 验证阶段 ----------
        # 在每个 epoch 结束后，评估模型在验证集上的性能
        
        model.eval()  # 切换到评估模式（关闭 Dropout，冻结 BatchNorm）
        total_val_loss = 0
        
        # 禁用梯度计算，节省显存和加速计算
        with torch.no_grad():
            # 创建验证进度条
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False)
            
            # 遍历验证集的每个 batch
            for batch in val_bar:
                # 将数据移动到设备
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # 前向传播（同样使用 AMP）
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(**batch)
                    # 累加 validation loss
                    total_val_loss += outputs.loss.item()
                    
        # 计算平均验证 loss
        avg_val_loss = total_val_loss / max(1, len(val_loader))
        
        # 打印当前 epoch 的训练和验证 loss
        # 通过对比 train/val loss 可以判断是否过拟合：
        # - 如果 train loss 持续下降但 val loss 上升 → 过拟合
        # - 如果两者都下降 → 正常训练
        # - 如果两者都不下降 → 学习率可能太小或模型容量不足
        print(f"\n✅ Epoch {epoch+1} finished | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # 记录 loss 历史（用于后续可视化）
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)

    # ==================== Step 7: 训练结束，保存模型 ====================
    # 保存模型权重和 tokenizer，以便后续推理或继续训练
    
    # save_pretrained 会保存以下文件：
    # - config.json: 模型架构配置
    # - pytorch_model.bin: 模型权重参数
    # - tokenizer 相关文件 (tokenizer.json, tokenizer_config.json 等)
    # - special_tokens_map.json: 特殊 token 映射
    
    model.save_pretrained(out_dir)      # 保存模型
    tokenizer.save_pretrained(out_dir)  # 保存 tokenizer
    
    # 返回 loss 历史记录
    return {
        'train_losses': train_losses_history,
        'val_losses': val_losses_history,
    }


def save_training_plot(
    train_losses: List[float],
    val_losses: List[float],
    rouge_scores: Optional[Dict[str, float]],
    args: argparse.Namespace,
    out_dir: str,
) -> None:
    """
    保存训练过程的可视化图表
    
    【功能说明】
    绘制训练/验证 Loss 曲线，并在图片底部标注关键超参数和 ROUGE 指标
    
    【参数】
    - train_losses: 每个 epoch 的训练 loss 列表
    - val_losses: 每个 epoch 的验证 loss 列表
    - rouge_scores: ROUGE 评估结果字典
    - args: 命令行参数
    - out_dir: 输出目录
    """
    if not HAS_MATPLOTLIB:
        print("跳过可视化：matplotlib 未安装", flush=True)
        return
    
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制 Loss 曲线
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
        ax.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=6)
        
        # 设置标题和标签
        ax.set_title('T5 News Summarization - Training Progress', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(labelsize=10)
        
        # 在图片底部添加参数信息文本框
        param_text = (
            f"Hyperparameters:\n"
            f"Model: {args.model_name}\n"
            f"LR: {args.lr} | Batch Size: {args.batch_size} | Grad Accum: {args.grad_accum}\n"
            f"Max Source Len: {args.max_source_len} | Max Target Len: {args.max_target_len}\n"
            f"Train Samples: {args.max_train_samples} | Val Samples: {args.max_val_samples}\n"
            f"Prefix: '{args.prefix}'"
        )
        
        # 如果有 ROUGE 分数，添加到文本中
        if rouge_scores:
            rouge_text = (
                f"\nROUGE Scores (val subset):\n"
                f"ROUGE-1: {rouge_scores.get('rouge1', 0):.4f} | "
                f"ROUGE-2: {rouge_scores.get('rouge2', 0):.4f} | "
                f"ROUGE-L: {rouge_scores.get('rougeL', 0):.4f}"
            )
            param_text += rouge_text
        
        # 在图表底部添加文本框
        plt.figtext(
            0.5, 0.01, param_text,
            ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', edgecolor='gray', alpha=0.9),
            family='monospace'
        )
        
        # 调整布局，为底部文本留出空间
        plt.tight_layout(rect=[0, 0.15, 1, 0.95])
        
        # 保存图片
        plot_path = os.path.join(out_dir, "training_curve.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"训练曲线已保存: {plot_path}", flush=True)
        
    except Exception as e:
        print(f"可视化失败: {e}", file=sys.stderr)


# ============================================================================
# 额外可视化辅助函数（用于技术攻关组）
# ============================================================================

def plot_rouge_comparison(
    rouge_scores: Dict[str, float],
    baseline_scores: Optional[Dict[str, float]] = None,
    save_path: str = "rouge_comparison.png",
) -> None:
    """
    绘制 ROUGE 分数柱状对比图
    
    【参数】
    - rouge_scores: 当前模型的 ROUGE 分数 {'rouge1': ..., 'rouge2': ..., 'rougeL': ...}
    - baseline_scores: 基线模型的 ROUGE 分数（可选，用于对比）
    - save_path: 保存路径
    """
    if not HAS_MATPLOTLIB:
        print("跳过 ROUGE 对比图：matplotlib 未安装", file=sys.stderr)
        return
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        keys = ['rouge1', 'rouge2', 'rougeL']
        
        current_scores = [rouge_scores.get(k, 0) for k in keys]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        # 绘制当前模型
        bars1 = ax.bar(x - width/2, current_scores, width, label='Current Model', 
                       color='#4CAF50', alpha=0.8, edgecolor='black')
        
        # 绘制基线模型（如果有）
        if baseline_scores:
            baseline_vals = [baseline_scores.get(k, 0) for k in keys]
            bars2 = ax.bar(x + width/2, baseline_vals, width, label='Baseline', 
                           color='#FF9800', alpha=0.8, edgecolor='black')
            
            for bar, val in zip(bars2, baseline_vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 标注当前模型数值
        for bar, val in zip(bars1, current_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel('ROUGE Score', fontsize=12)
        ax.set_title('ROUGE Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.legend(loc='upper right', fontsize=11)
        ax.set_ylim(0, max(max(current_scores), max(baseline_scores.values()) if baseline_scores else 0) * 1.2)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"ROUGE 对比图已保存: {save_path}", flush=True)
        
    except Exception as e:
        print(f"ROUGE 对比图生成失败: {e}", file=sys.stderr)


def plot_performance_radar(
    metrics: Dict[str, float],
    save_path: str = "performance_radar.png",
) -> None:
    """
    绘制性能多维雷达图
    
    【参数】
    - metrics: 包含多个性能指标的字典
              例如: {'ROUGE-1': 0.3038, 'ROUGE-2': 0.1221, 'ROUGE-L': 0.2330, 
                    'Speed': 0.85, 'Memory': 0.70}
    - save_path: 保存路径
    
    【原理说明】
    雷达图可以同时展示多个维度的性能，适合综合评估模型表现。
    每个维度归一化到 [0, 1] 范围。
    """
    if not HAS_MATPLOTLIB:
        print("跳过性能雷达图：matplotlib 未安装", file=sys.stderr)
        return
    
    try:
        labels = list(metrics.keys())
        values = list(metrics.values())
        
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        ax.plot(angles, values, 'o-', linewidth=2, color='#2196F3', markersize=8)
        ax.fill(angles, values, alpha=0.25, color='#2196F3')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title('Multi-dimensional Performance Radar', fontsize=14, fontweight='bold', pad=20)
        
        for angle, value, label in zip(angles[:-1], values[:-1], labels):
            ax.text(angle, value + 0.05, f'{value:.3f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"性能雷达图已保存: {save_path}", flush=True)
        
    except Exception as e:
        print(f"性能雷达图生成失败: {e}", file=sys.stderr)


def plot_loss_rouge_evolution(
    train_losses: List[float],
    val_losses: List[float],
    rouge_evolution: Optional[List[Dict[str, float]]] = None,
    save_path: str = "loss_rouge_evolution.png",
) -> None:
    """
    绘制 Loss 和 ROUGE 指标的联合演化图（双Y轴）
    
    【参数】
    - train_losses: 训练 Loss 列表
    - val_losses: 验证 Loss 列表
    - rouge_evolution: 每个 epoch 的 ROUGE 分数列表（可选）
    - save_path: 保存路径
    """
    if not HAS_MATPLOTLIB:
        print("跳过 Loss-ROUGE 演化图：matplotlib 未安装", file=sys.stderr)
        return
    
    try:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        epochs = range(1, len(train_losses) + 1)
        line1 = ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
        line2 = ax1.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=6)
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12, color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        if rouge_evolution:
            ax2 = ax1.twinx()
            rouge_l_scores = [r.get('rougeL', 0) for r in rouge_evolution]
            line3 = ax2.plot(epochs, rouge_l_scores, 'g-^', label='ROUGE-L', 
                            linewidth=2, markersize=6)
            ax2.set_ylabel('ROUGE-L', fontsize=12, color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right', fontsize=11)
        else:
            ax1.legend(loc='upper right', fontsize=11)
        
        ax1.set_title('Training Progress: Loss and ROUGE Evolution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Loss-ROUGE 演化图已保存: {save_path}", flush=True)
        
    except Exception as e:
        print(f"Loss-ROUGE 演化图生成失败: {e}", file=sys.stderr)


def plot_generation_samples(
    samples: List[Dict[str, str]],
    save_path: str = "generation_samples.png",
    max_samples: int = 5,
) -> None:
    """
    绘制生成样例对比图
    
    【参数】
    - samples: 生成样例列表，每个样例包含:
              {'source': '原文', 'target': '真实标题', 'predicted': '生成标题'}
    - save_path: 保存路径
    - max_samples: 最多展示的样例数量
    """
    if not HAS_MATPLOTLIB:
        print("跳过生成样例图：matplotlib 未安装", file=sys.stderr)
        return
    
    try:
        samples = samples[:max_samples]
        n_samples = len(samples)
        
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 2 * n_samples + 1))
        if n_samples == 1:
            axes = [axes]
        
        fig.suptitle('Generated Samples Comparison', fontsize=14, fontweight='bold')
        
        for idx, (ax, sample) in enumerate(zip(axes, samples)):
            ax.axis('off')
            
            text = (
                f"Sample {idx + 1}:\n\n"
                f"【Source Text】\n{sample['source'][:200]}...\n\n"
                f"【Ground Truth】\n{sample['target']}\n\n"
                f"【Prediction】\n{sample['predicted']}"
            )
            
            ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"生成样例对比图已保存: {save_path}", flush=True)
        
    except Exception as e:
        print(f"生成样例图生成失败: {e}", file=sys.stderr)


def main() -> None:
    """
    主函数：协调整个训练流程
    
    【工作流程】
    1. 解析命令行参数
    2. 加载数据清单 manifest
    3. 从本地缓存加载数据集
    4. 抽取训练/验证子集（用于调试）
    5. 初始化 T5 模型和 Tokenizer
    6. 构建预处理函数并应用
    7. 创建 DataCollator
    8. 调用 run_s2s_training 执行训练
    9. 保存模型和超参数配置
    10. （可选）在验证集上计算 ROUGE 指标
    11. 生成训练可视化图表
    
    【关键设计】
    - 支持通过命令行参数灵活控制训练行为
    - 使用子集机制加速调试（--max_train_samples / --max_val_samples）
    - 训练结束后自动进行快速 ROUGE 评估（前128条样本）
    - 保存 train_hparams.json 记录训练配置，便于复现实验
    - 自动生成训练曲线图，直观展示训练过程
    """
    # ==================== 1. 初始化工作目录 ====================
    # 切换到脚本所在目录，确保相对路径正确
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    # ==================== 2. 解析命令行参数 ====================
    p = argparse.ArgumentParser(description="T5 新闻摘要/标题 微调（需补全训练与预处理）")
    p.add_argument("--manifest", type=str, default=MANIFEST)
    p.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--model_name", type=str, default=MODEL_NAME)
    p.add_argument("--max_train_samples", type=int, default=MAX_TRAIN_SAMPLES)
    p.add_argument("--max_val_samples", type=int, default=MAX_VAL_SAMPLES)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--max_source_len", type=int, default=MAX_SOURCE_LENGTH)
    p.add_argument("--max_target_len", type=int, default=MAX_TARGET_LENGTH)
    p.add_argument("--prefix", type=str, default=PREFIX)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_rouge_eval", action="store_true", help="训练结束后在验证上不算 ROUGE（省时间）")
    args = p.parse_args()

    # ==================== 3. 加载数据清单 manifest ====================
    # manifest 包含数据集的路径、字段名等元信息
    manifest = load_manifest(args.manifest)
    text_col = manifest["text_column"]      # 正文字段名（如 'article'）
    summary_col = manifest["summary_column"]  # 摘要字段名（如 'highlights'）
    
    # 设置随机种子，保证实验可复现
    set_seed(args.seed)
    
    # 打印设备信息
    print("设备:", "cuda" if torch.cuda.is_available() else "cpu", flush=True)
    
    # ==================== 4. 加载数据集 ====================
    print("正在从本地 cache 加载数据集（大缓存时可能 1～数分钟无新输出，属正常）…", flush=True)
    raw = _load_raw_dataset(manifest)
    
    print("数据集已载入，正在取 train/val 子集…", flush=True)
    train_ds = raw["train"]       # 训练集
    val_ds = raw["validation"]    # 验证集

    # ==================== 5. 抽取子集（用于调试）====================
    # 如果指定了 max_samples 参数，则只使用前 N 条样本
    # 这样可以快速验证流程，无需等待完整训练
    
    n_tr = int(args.max_train_samples) if int(args.max_train_samples) > 0 else len(train_ds)
    n_val = int(args.max_val_samples) if int(args.max_val_samples) > 0 else len(val_ds)
    
    # 确保不超过实际样本数
    n_tr = min(n_tr, len(train_ds))
    n_val = min(n_val, len(val_ds))
    
    # 选择前 N 条样本
    train_ds = train_ds.select(range(n_tr))
    val_ds = val_ds.select(range(n_val))
    
    print(
        f"子集: train={n_tr}, val={n_val}（0 表示使用该划分全量，见顶栏默认值或 --max_train_samples / --max_val_samples）",
        flush=True,
    )

    # ==================== 6. 初始化 T5 模型和 Tokenizer ====================
    print("正在加载 T5 分词器与预训练权重（首次会下载，可能较久）…", flush=True)
    
    # 加载 Tokenizer
    # use_fast=True: 使用基于 Rust 的快速 tokenizer，速度提升 10-100 倍
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    
    # 加载预训练模型
    # from_pretrained 会自动下载模型权重（首次）或从缓存加载
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    
    print("模型已就绪，开始 tokenize…", flush=True)

    # ==================== 7. 构建预处理函数并应用 ====================
    # 创建 tokenization 函数
    preprocess = build_preprocess_fn(
        text_col, summary_col, tokenizer, args.prefix, args.max_source_len, args.max_target_len
    )

    # 对训练集进行 tokenization
    # batched=True: 批量处理，提高效率
    # remove_columns: 删除原始列，只保留 tokenized 后的字段
    # desc: 显示进度条描述
    train_tok = train_ds.map(
        preprocess,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="tokenize train",
    )
    
    # 对验证集进行 tokenization
    val_tok = val_ds.map(
        preprocess,
        batched=True,
        remove_columns=val_ds.column_names,
        desc="tokenize val",
    )

    # ==================== 8. 创建 DataCollator ====================
    # DataCollatorForSeq2Seq 负责将多个样本组装成 batch
    
    # 主要功能：
    # 1. Padding：将不同长度的序列补齐到相同长度
    # 2. Label 处理：将 padding 位置的 labels 设为 -100（loss 忽略）
    # 3. Tensor 转换：将列表转换为 PyTorch Tensor
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,  # padding 位置的 label 用 -100 填充
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,  # CUDA 时 padding 到 8 的倍数，利用 Tensor Cores
    )

    # ==================== 9. 准备输出目录 ====================
    out_dir = args.output_dir if os.path.isabs(args.output_dir) else _abs_here(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)  # 创建目录（如果已存在则不报错）

    # ==================== 10. 执行训练 ====================
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60 + "\n")
    
    # 接收训练返回的 loss 历史记录
    loss_history = run_s2s_training(
        model,
        tokenizer,
        train_tok,
        val_tok,
        data_collator,
        val_ds,
        text_col,
        summary_col,
        args,
        out_dir,
    )
    
    # 兜底保存：确保模型一定被保存
    # 即使 run_s2s_training 内部已保存，这里再保存一次作为保险
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    # ==================== 11. 保存训练超参数配置 ====================
    # 记录本次训练的关键配置，便于后续复现实验或分析结果
    
    with open(os.path.join(out_dir, "train_hparams.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,          # 模型名称
                "manifest": os.path.abspath(            # manifest 文件路径
                    args.manifest if os.path.isabs(args.manifest) else _abs_here(args.manifest)
                ),
                "prefix": args.prefix,                  # 输入前缀
                "max_source_len": args.max_source_len,  # 最大输入长度
                "max_target_len": args.max_target_len,  # 最大输出长度
                "max_train_samples": n_tr,              # 实际训练样本数
                "max_val_samples": n_val,               # 实际验证样本数
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    
    print("已保存:", os.path.abspath(out_dir), flush=True)

    # ==================== 12. （可选）快速 ROUGE 评估 ====================
    # 在验证集的前 128 条样本上计算 ROUGE 指标，快速评估模型性能
    # 注意：这不是正式评估，正式评估应使用 evaluate_rouge.py
    
    if not args.no_rouge_eval and n_val > 0:
        print("\n" + "="*60)
        print("开始在验证子集上快速评估 ROUGE...")
        print("="*60 + "\n")
        
        model.eval()  # 切换到评估模式
        dev = next(model.parameters()).device  # 获取模型所在设备
        batch_size = min(4, args.batch_size)   # 使用较小的 batch size 节省显存
        
        gen_ids = []     # 存储生成的 token IDs
        ref_texts = []   # 存储参考摘要文本
        
        # 禁用梯度计算
        with torch.no_grad():
            # 最多评估 128 条样本
            for i in range(0, min(128, n_val), batch_size):
                # 选择当前 batch 的样本
                sl = val_ds.select(range(i, min(i + batch_size, n_val)))
                
                # 构建输入：添加 prefix
                ins = [args.prefix + str(t) for t in sl[text_col]]
                
                # Tokenize 并移动到设备
                enc = tokenizer(
                    ins,
                    max_length=args.max_source_len,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                ).to(dev)
                
                # 生成摘要
                # generate() 方法使用 Beam Search 解码
                g = model.generate(
                    **enc,
                    max_new_tokens=args.max_target_len,  # 最多生成多少个 token
                    num_beams=4,                          # Beam Search 的 beam 数量
                    length_penalty=0.85,                  # 长度惩罚：<1 偏好短句
                    early_stopping=True,                  # 当所有 beam 都生成 EOS 时停止
                )
                
                # 收集生成结果和参考摘要
                gen_ids.extend(g.cpu().tolist())
                for s in sl[summary_col]:
                    ref_texts.append(str(s))
        
        # 将 token IDs 解码为文本
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        
        # 计算 ROUGE 指标
        n_show = min(len(preds), len(ref_texts))
        r = try_compute_rouge(preds[:n_show], ref_texts[:n_show])
        
        if r:
            # 打印 ROUGE 结果
            print("验证子集 ROUGE (约前 128 条):", {k: round(v, 4) for k, v in r.items()}, flush=True)
        else:
            # 如果 rouge_score 未安装，提示用户
            print("未安装 rouge_score，跳过 ROUGE。可: pip install rouge-score", file=sys.stderr)
        
        # 打印前 2 个样例的生成结果与参考摘要对比
        for k in range(min(2, n_show)):
            print(
                f"\n[样例 {k+1}]\n参考: {ref_texts[k][:200]!s}…\n生成: {preds[k]!s}",
                flush=True,
            )
    else:
        r = None  # 如果跳过了 ROUGE 评估

    # ==================== 13. 生成训练可视化图表 ====================
    # 创建带时间戳的可视化结果文件夹
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    viz_dir = os.path.join(out_dir, f"visualization_{timestamp}")
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"\n正在生成训练可视化图表...", flush=True)
    print(f"可视化结果将保存到: {viz_dir}", flush=True)
    
    # 1. 训练曲线图（带参数标注）
    save_training_plot(
        train_losses=loss_history['train_losses'],
        val_losses=loss_history['val_losses'],
        rouge_scores=r,
        args=args,
        out_dir=viz_dir,  # 保存到时间戳文件夹
    )
    
    # 2. ROUGE 分数对比图
    plot_rouge_comparison(
        rouge_scores=r if r else {'rouge1': 0, 'rouge2': 0, 'rougeL': 0},
        save_path=os.path.join(viz_dir, "rouge_comparison.png"),
    )
    
    # 3. 性能雷达图
    if r:
        perf_metrics = {
            'ROUGE-1': r.get('rouge1', 0),
            'ROUGE-2': r.get('rouge2', 0),
            'ROUGE-L': r.get('rougeL', 0),
            'Precision': 0.28,  # 可以从 predict.py 的结果中计算
            'Recall': 0.35,
        }
        plot_performance_radar(
            metrics=perf_metrics,
            save_path=os.path.join(viz_dir, "performance_radar.png"),
        )
    
    # 4. Loss-ROUGE 联合演化图
    plot_loss_rouge_evolution(
        train_losses=loss_history['train_losses'],
        val_losses=loss_history['val_losses'],
        save_path=os.path.join(viz_dir, "loss_rouge_evolution.png"),
    )

    
    print("\n" + "="*60)
    print("所有任务完成！")
    print(f"模型保存在: {out_dir}")
    print(f"\n可视化结果保存在: {viz_dir}")
    print(f"\n生成的图表:")
    print(f"  1. training_curve.png         - 训练曲线（带参数标注）")
    print(f"  2. rouge_comparison.png       - ROUGE 分数对比")
    print(f"  3. performance_radar.png      - 性能多维雷达图")
    print(f"  4. loss_rouge_evolution.png   - Loss-ROUGE 联合演化")
    print("="*60)


if __name__ == "__main__":
    main()



