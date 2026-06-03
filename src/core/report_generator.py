#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测报告生成模块
职责：将 ROUGE 评测结果保存为 JSON、Markdown 和文本报告
"""
import os
import json
from typing import Any, Dict, List, Optional
from datetime import datetime


def save_rouge_json(
    rouge_scores: Dict[str, float],
    exp_id: str,
    split: str,
    output_path: str,
) -> None:
    """
    保存 ROUGE 结果为 JSON 格式
    
    Args:
        rouge_scores: ROUGE 分数字典
        exp_id: 实验ID
        split: 数据集划分（validation/test）
        output_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    result = {
        "exp_id": exp_id,
        "split": split,
        "n_samples": rouge_scores.get("n", 0),
        "rouge1": round(rouge_scores.get("rouge1", 0.0), 4),
        "rouge2": round(rouge_scores.get("rouge2", 0.0), 4),
        "rougeL": round(rouge_scores.get("rougeL", 0.0), 4),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"✅ ROUGE 结果已写入: {output_path}")


def save_examples_markdown(
    preds: List[str],
    refs: List[str],
    rouge_scores: Dict[str, float],
    exp_id: str,
    output_path: str,
    max_examples: int = 10,
) -> None:
    """
    保存生成样例为 Markdown 格式（符合评测要求）
    
    Args:
        preds: 预测列表
        refs: 参考列表
        rouge_scores: ROUGE 分数
        exp_id: 实验ID
        output_path: 输出文件路径
        max_examples: 最大样例数
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        # 标题
        f.write(f"# ROUGE 评测样例 - {exp_id}\n\n")
        
        # 评测信息
        f.write(f"**评测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**ROUGE-1**: {rouge_scores.get('rouge1', 0.0):.4f}  \n")
        f.write(f"**ROUGE-2**: {rouge_scores.get('rouge2', 0.0):.4f}  \n")
        f.write(f"**ROUGE-L**: {rouge_scores.get('rougeL', 0.0):.4f}  \n")
        f.write(f"**评测样本数**: {rouge_scores.get('n', 0)}\n\n")
        
        f.write("---\n\n")
        
        # 样例详情
        n_show = min(max_examples, len(preds))
        f.write(f"## 生成样例（前 {n_show} 条）\n\n")
        
        for k in range(n_show):
            f.write(f"### 样例 {k+1}\n\n")
            f.write(f"**参考摘要**:\n> {refs[k]}\n\n")
            f.write(f"**模型生成**:\n> {preds[k]}\n\n")
            f.write("---\n\n")
    
    print(f"✅ 生成样例已写入: {output_path}")


def save_full_report(
    rouge_scores: Dict[str, float],
    preds: List[str],
    refs: List[str],
    exp_id: str,
    split: str,
    ckpt_path: str,
    output_path: str,
    max_examples: int = 5,
) -> None:
    """
    保存完整评测报告（文本格式）
    
    Args:
        rouge_scores: ROUGE 分数
        preds: 预测列表
        refs: 参考列表
        exp_id: 实验ID
        split: 数据集划分
        ckpt_path: checkpoint 路径
        output_path: 输出文件路径
        max_examples: 最大样例数
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        # 报告头部
        f.write("=" * 70 + "\n")
        f.write(f"ROUGE 评测报告 - {exp_id}\n")
        f.write("=" * 70 + "\n\n")
        
        # 基本信息
        f.write(f"评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"数据集划分: {split}\n")
        f.write(f"评测样本数: {rouge_scores.get('n', 0)}\n\n")
        
        # ROUGE 结果
        f.write("-" * 70 + "\n")
        f.write("ROUGE 指标\n")
        f.write("-" * 70 + "\n")
        f.write(f"  ROUGE-1: {rouge_scores.get('rouge1', 0.0):.4f}\n")
        f.write(f"  ROUGE-2: {rouge_scores.get('rouge2', 0.0):.4f}\n")
        f.write(f"  ROUGE-L: {rouge_scores.get('rougeL', 0.0):.4f}\n\n")
        
        # 生成样例
        f.write("-" * 70 + "\n")
        f.write(f"生成样例（前 {min(max_examples, len(preds))} 条）\n")
        f.write("-" * 70 + "\n\n")
        
        n_show = min(max_examples, len(preds))
        for k in range(n_show):
            f.write(f"[样例 {k+1}]\n")
            f.write(f"参考: {refs[k]}\n\n")
            f.write(f"生成: {preds[k]}\n\n")
            f.write("-" * 70 + "\n\n")
        
        # 报告尾部
        f.write("\n" + "=" * 70 + "\n")
        f.write("报告结束\n")
        f.write("=" * 70 + "\n")
    
    print(f"✅ 完整报告已写入: {output_path}")


def generate_all_reports(
    rouge_scores: Dict[str, float],
    preds: List[str],
    refs: List[str],
    exp_id: str,
    split: str,
    ckpt_path: str,
    project_root: str,
) -> Dict[str, str]:
    """
    生成所有报告文件（JSON + Markdown + 文本）
    
    Args:
        rouge_scores: ROUGE 分数
        preds: 预测列表
        refs: 参考列表
        exp_id: 实验ID
        split: 数据集划分
        ckpt_path: checkpoint 路径
        project_root: 项目根目录
        
    Returns:
        生成的文件路径字典
    """
    # ✅ 使用模型名 + 样本数命名，避免覆盖
    n_samples = rouge_scores.get("n", 0)
    file_suffix = f"{exp_id}_n{n_samples}"
    
    # 1. JSON 结果
    json_path = os.path.join(project_root, "results", f"{file_suffix}_rouge.json")
    save_rouge_json(rouge_scores, exp_id, split, json_path)
    
    # 2. Markdown 样例（符合评测要求的 examples/outputs.md）
    # ✅ 固定文件名，每次覆盖（保持最新结果）
    md_path = os.path.join(project_root, "examples", "outputs.md")
    save_examples_markdown(preds, refs, rouge_scores, exp_id, md_path)
    
    # 3. 完整文本报告
    txt_path = os.path.join(project_root, "results", f"{file_suffix}_report.txt")
    save_full_report(rouge_scores, preds, refs, exp_id, split, ckpt_path, txt_path)
    
    return {
        "json": json_path,
        "markdown": md_path,
        "text": txt_path,
    }
