#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多种子消融实验流水线
=====================
对每个消融实验配置，使用多个随机种子（默认 5 个）分别执行 训练 → 评测，
汇总均值 ± 标准差，生成具有统计意义的评测报告。

用法：
    python run_pipeline.py                          # 运行所有消融实验
    python run_pipeline.py --seeds 5                 # 自定义种子数
    python run_pipeline.py --skip_trained            # 跳过已有训练的，只做评测

设计要点：
    - 广度优先遍历所有 YAML 配置
    - 深度验证：每个配置跑 N 个种子，计算 Mean ± Std
    - 若某个种子失败，不影响其他种子，最终结果基于成功数
    - 最终生成统计报告，结论可被复现
"""
import os
import sys
import json
import glob
import time
import logging
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.configs.config_manager import load_config, load_full_config


# ==========================================
# 全局配置
# ==========================================

DEFAULT_SEEDS = [42, 123, 999, 2024, 7777]   # 固定种子列表，确保可复现
# DEFAULT_SEEDS = [42]
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
TRAIN_SCRIPT = os.path.join("scripts", "train.py")
EVAL_SCRIPT = os.path.join("scripts", "evaluate.py")
ABLATION_DIR = "src/configs/ablation"

# ==========================================
# 日志
# ==========================================

log_file = f"pipeline_{time.strftime('%Y%m%d_%H%M')}.log"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


# ==========================================
# 1. 加载消融实验矩阵
# ==========================================

def load_ablation_experiments(config_dir: str = ABLATION_DIR) -> List[dict]:
    """扫描 configs/ablation/ 目录，加载所有实验配置"""
    yaml_files = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))

    if not yaml_files:
        logging.warning(f"⚠️ 未在 {config_dir}/ 找到任何 YAML 配置文件！")
        return []

    experiments = []
    for yaml_path in yaml_files:
        # 跳过 baseline.yaml（作为父类，不直接运行）
        if "baseline" in os.path.basename(yaml_path).lower():
            continue

        try:
            config = load_config(yaml_path)
            exp = {
                "id": config.id,
                "config_file": os.path.basename(yaml_path),
                "lr": config.training.lr,
                "bs": config.training.batch_size,
                "ga": config.training.grad_accum,
                "epochs": config.training.epochs,
                "samples": config.data.max_train_samples,
                "skip_training": config.training.skip_training,
            }
            experiments.append(exp)
            logging.info(f"📋 发现实验 [{exp['id']}] <- {exp['config_file']}")
        except Exception as e:
            logging.error(f"❌ 加载 {yaml_path} 失败: {e}")
            continue

    return experiments


# ==========================================
# 2. 命令执行
# ==========================================

def run_command(cmd: str, desc: str, timeout: int = 7200) -> bool:
    """执行 shell 命令，返回是否成功"""
    logging.info(f"\n{'=' * 60}")
    logging.info(f"⚙️  {desc}")
    logging.info(f"💻 {cmd}")
    logging.info(f"{'=' * 60}\n")
    try:
        subprocess.run(cmd, shell=True, check=True, timeout=timeout)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ {desc} 失败 (returncode={e.returncode})")
        return False
    except subprocess.TimeoutExpired:
        logging.error(f"⏰ {desc} 超时 ({timeout}s)")
        return False


# ==========================================
# 3. 对单个实验×单个种子 执行训练+评测
# ==========================================

def run_single_seed(
    exp_id: str,
    config_file: str,
    seed: int,
    output_base: str,
    results_dir: str,
    eval_max_samples: int,
    skip_train: bool = False,
    with_bertscore: bool = False,
    skip_training: bool = False,  # 来自配置，model-level 跳过训练
) -> Optional[Dict[str, Any]]:
    """
    对单个实验、单个种子运行 训练 → 评测

    Args:
        exp_id: 实验 ID
        config_file: 相对路径的 YAML 文件名
        seed: 随机种子
        output_base: checkpoint 根目录
        results_dir: 结果存放目录
        eval_max_samples: 评测样本数
        skip_train: 若为 True，跳过训练步骤（只评测）
        with_bertscore: 是否启用 BERTScore 语义相似度

    Returns:
        指标 dict，若失败则返回 None
    """
    seed_tag = f"{exp_id}_seed{seed}"
    ckpt_path = os.path.join(output_base, seed_tag)

    # ── 训练 ──
    if skip_training:
        # model-level 跳过训练：只保存预训练权重
        train_cmd = (
            f"python {TRAIN_SCRIPT} "
            f"--config {os.path.join(ABLATION_DIR, config_file)} "
            f"--seed {seed} "
            f"--output_dir {output_base} "
            f"--exp_id {seed_tag} "
            f"--no_rouge_eval "
            f"--skip_training "
        )
        if not run_command(train_cmd, f"保存预训练权重 [{exp_id}] seed={seed}"):
            return None
    elif not skip_train:
        train_cmd = (
            f"python {TRAIN_SCRIPT} "
            f"--config {os.path.join(ABLATION_DIR, config_file)} "
            f"--seed {seed} "
            f"--output_dir {output_base} "
            f"--exp_id {seed_tag} "
            f"--no_rouge_eval "  # 跳过训练中的快速 ROUGE，用 evaluate.py 统一评测
        )
        if not run_command(train_cmd, f"训练 [{exp_id}] seed={seed}"):
            return None
    else:
        if os.path.isdir(ckpt_path) and os.path.isfile(os.path.join(ckpt_path, "config.json")):
            logging.info(f"⏭️  跳过训练：{ckpt_path} 已存在")
        else:
            logging.warning(f"⚠️  跳过训练但 {ckpt_path} 不存在，仍尝试评测...")

    # ── 评测 ──
    seed_json = os.path.join(results_dir, f"seed{seed}.json")
    os.makedirs(os.path.dirname(seed_json), exist_ok=True)

    eval_cmd = (
        f"python {EVAL_SCRIPT} "
        f"--ckpt {ckpt_path} "
        f"--seed {seed} "
        f"--max_samples {eval_max_samples} "
        f"--output_json {seed_json} "
    )
    if with_bertscore:
        eval_cmd += "--with_bertscore "
    if not run_command(eval_cmd, f"评测 [{exp_id}] seed={seed}"):
        return None

    # ── 读取评测结果 ──
    try:
        with open(seed_json, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        logging.info(f"✅ [{exp_id}] seed={seed}: "
                      f"ROUGE-1={metrics.get('rouge1', 'N/A')}, "
                      f"ROUGE-2={metrics.get('rouge2', 'N/A')}, "
                      f"ROUGE-L={metrics.get('rougeL', 'N/A')}")
        return metrics
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logging.error(f"❌ 读取 {seed_json} 失败: {e}")
        return None


# ==========================================
# 4. 统计聚合
# ==========================================

def aggregate_metrics(per_seed_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    对多次种子运行的结果进行汇总统计

    对每个数值型指标计算：
        - mean: 均值
        - std:  标准差
        - min:  最小值
        - max:  最大值
        - n:    有效种子数

    Returns:
        包含统计结果的 dict
    """
    if not per_seed_metrics:
        return {}

    # 收集所有数值型指标
    numeric_keys: Dict[str, List[float]] = {}
    for metrics in per_seed_metrics:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                numeric_keys.setdefault(k, []).append(float(v))

    result = {}
    for k, values in sorted(numeric_keys.items()):
        arr = np.array(values)
        result[f"{k}_mean"] = float(np.mean(arr))
        result[f"{k}_std"] = float(np.std(arr, ddof=1))  # 样本标准差
        result[f"{k}_min"] = float(np.min(arr))
        result[f"{k}_max"] = float(np.max(arr))
    result["n_seeds"] = len(per_seed_metrics)

    return result


def format_metric_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    """格式化 `均值 ± 标准差` 字符串"""
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


# ==========================================
# 5. 生成统计报告
# ==========================================

def generate_ablation_report(
    all_experiments: List[Dict],
    aggregated_results: Dict[str, Dict[str, Any]],
    per_seed_raw: Dict[str, List[Dict[str, Any]]],
    output_dir: str,
):
    """
    生成多维度消融实验统计报告（Markdown 格式）

    Args:
        all_experiments: 实验配置列表
        aggregated_results: exp_id -> 聚合指标
        per_seed_raw: exp_id -> 每个种子的原始指标列表
        output_dir: 报告输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "ablation_statistical_report.md")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(report_path, "w", encoding="utf-8") as f:
        # ── 标题 ──
        f.write("# 消融实验统计报告\n\n")
        f.write(f"**生成时间**: {timestamp}  \n")
        f.write(f"**每实验种子数**: 各实验有效种子数见下方表格\n\n")

        # ── 实验配置总览 ──
        f.write("---\n\n")
        f.write("## 实验配置总览\n\n")
        f.write("| 实验 ID | 配置文件 | 学习率 | Batch | GradAccum | Epochs | 训练样本 |\n")
        f.write("|---------|----------|--------|-------|-----------|--------|----------|\n")
        for exp in all_experiments:
            cfg = exp["config_file"].replace("_", "\\_")
            f.write(f"| {exp['id']} | {cfg} | {exp['lr']} | {exp['bs']} | "
                    f"{exp['ga']} | {exp['epochs']} | {exp['samples']} |\n")

        # ── 核心指标对比表 ──
        f.write("\n---\n\n")
        f.write("## 核心指标对比（均值 ± 标准差）\n\n")

        # 收集所有实验中出现的指标名称（ROUGE + 额外指标如 bertscore_f1）
        all_metric_bases: List[str] = []
        for exp in all_experiments:
            agg = aggregated_results.get(exp["id"], {})
            for k in agg:
                if k.endswith("_mean") and not k.startswith("_"):
                    base = k.rsplit("_", 1)[0]
                    if base not in all_metric_bases:
                        all_metric_bases.append(base)
        # 按 ROUGE -> BERTScore -> 其他 排序
        metric_order = {
            "rouge1": 0, "rouge2": 1, "rougeL": 2,
            "bertscore_f1": 10, "bertscore_precision": 11, "bertscore_recall": 12,
        }
        all_metric_bases.sort(key=lambda x: metric_order.get(x, 99))

        # 表头
        header = "| 实验 ID | 有效种子 |" + "".join(f" {m.upper()} |" for m in all_metric_bases)
        sep = "|---------|----------|" + "|".join("---------" for _ in all_metric_bases) + "|"
        f.write(header + "\n")
        f.write(sep + "\n")

        table_rows = []
        for exp in all_experiments:
            eid = exp["id"]
            agg = aggregated_results.get(eid, {})
            if not agg:
                continue
            n = agg.get("n_seeds", 0)
            cells = []
            for base in all_metric_bases:
                mn = agg.get(f"{base}_mean", None)
                sd = agg.get(f"{base}_std", None)
                if mn is not None and sd is not None:
                    cells.append(format_metric_mean_std(mn, sd))
                else:
                    cells.append("—")
            sort_key = agg.get(f"{all_metric_bases[0]}_mean", 0) if all_metric_bases else 0
            table_rows.append((sort_key, eid, n, cells))

        # 按首个指标降序排列
        table_rows.sort(key=lambda x: x[0], reverse=True)

        for _, eid, n, cells in table_rows:
            f.write(f"| {eid} | {n} | " + " | ".join(cells) + " |\n")

        # ── 详细结果（含额外指标） ──
        f.write("\n---\n\n")
        f.write("## 各实验详细结果\n\n")

        for exp in all_experiments:
            eid = exp["id"]
            agg = aggregated_results.get(eid, {})
            raw_list = per_seed_raw.get(eid, [])
            if not agg:
                f.write(f"### {eid}\n> ⚠️ 无有效数据\n\n")
                continue

            f.write(f"### {eid}\n\n")
            f.write(f"- 配置文件: `{exp['config_file']}`\n")
            f.write(f"- 有效种子数: {agg['n_seeds']}\n\n")

            # 所有指标汇总
            f.write("| 指标 | 均值 ± 标准差 | 最小值 | 最大值 |\n")
            f.write("|------|--------------|--------|--------|\n")

            for k in sorted(agg.keys()):
                if k == "n_seeds":
                    continue
                # 解析键名: rouge1_mean, rouge1_std, rouge1_min, rouge1_max
                parts = k.rsplit("_", 1)
                if len(parts) != 2:
                    continue
                base_name, stat_type = parts
                if stat_type == "mean":
                    std_key = f"{base_name}_std"
                    min_key = f"{base_name}_min"
                    max_key = f"{base_name}_max"
                    mn = agg.get(k, 0)
                    sd = agg.get(std_key, 0)
                    mi = agg.get(min_key, 0)
                    mx = agg.get(max_key, 0)
                    f.write(f"| {base_name} | {format_metric_mean_std(mn, sd)} | "
                            f"{mi:.4f} | {mx:.4f} |\n")

            # 每次种子的原始数据
            f.write("\n**各种子原始值**:\n\n")
            for r in raw_list:
                seed_val = r.get("seed", "?")
                vals = ", ".join(
                    f"{k}={v:.4f}" for k, v in sorted(r.items())
                    if isinstance(v, (int, float)) and k != "n_samples"
                )
                f.write(f"- seed={seed_val}: {vals}\n")

            f.write("\n")

        # ── 结论与建议 ──
        f.write("\n---\n\n")
        f.write("## 统计摘要\n\n")
        f.write("- 每实验运行 N 个随机种子（默认 5 个），结果以均值 ± 样本标准差呈现\n")
        f.write("- 标准差越小，说明该实验配置对初始化/抽样随机性越不敏感\n")
        f.write("- 若某实验标准差显著大于其他实验，说明该配置**不稳定**，结论需谨慎\n")
        f.write(f"- 所有结果均保存于: {output_dir}/\n")

    logging.info(f"📄 统计报告已生成: {report_path}")
    return report_path


# ==========================================
# 6. 主流程
# ==========================================

def main():
    """多种子消融实验流水线主函数"""
    logging.info("🚀 T5-News 多种子消融实验流水线启动！")
    logging.info("=" * 60)
    logging.info(f"设计：广度 × 深度遍历 | 每实验 {len(DEFAULT_SEEDS)} 种随机种子")
    logging.info("=" * 60)
    start_time = time.time()

    # ── 加载全局配置 ──
    cfg = load_full_config()
    OUTPUT_BASE = cfg.paths.ablation_base     # checkpoint 输出根目录
    EVAL_MAX_SAMPLES = cfg.evaluation.max_samples
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    logging.info(f"📂 Checkpoint 目录: {OUTPUT_BASE}")
    logging.info(f"📂 评测结果目录: {RESULTS_DIR}")
    logging.info(f"🔬 评测样本数: {EVAL_MAX_SAMPLES}")
    logging.info(f"🎲 种子列表: {DEFAULT_SEEDS}")

    # ── 确保数据就绪 ──
    if not run_command("python scripts/prepare_data.py", "初始化数据管道"):
        logging.error("❌ 数据准备失败，流水线中止")
        return

    # ── 加载实验配置 ──
    experiments = load_ablation_experiments()
    if not experiments:
        logging.error("❌ 未找到任何实验配置，流水线中止")
        return

    logging.info(f"\n📊 共 {len(experiments)} 组实验 × {len(DEFAULT_SEEDS)} 种种子 = "
                  f"{len(experiments) * len(DEFAULT_SEEDS)} 次运行\n")

    # ── 逐实验遍历 ──
    aggregated_results: Dict[str, Dict[str, Any]] = {}
    per_seed_raw: Dict[str, List[Dict[str, Any]]] = {}

    for exp_idx, exp in enumerate(experiments, 1):
        eid = exp["id"]
        exp_results_dir = os.path.join(RESULTS_DIR, eid)
        per_seed_raw[eid] = []

        logging.info(f"\n{'⭐' * 40}")
        logging.info(f"[{exp_idx}/{len(experiments)}] 实验: {eid} ({exp['config_file']})")
        logging.info(f"{'⭐' * 40}")

        for seed in DEFAULT_SEEDS:
            logging.info(f"  ── 种子 {seed} ──")
            metrics = run_single_seed(
                exp_id=eid,
                config_file=exp["config_file"],
                seed=seed,
                output_base=OUTPUT_BASE,
                results_dir=exp_results_dir,
                eval_max_samples=EVAL_MAX_SAMPLES,
                with_bertscore=True,
                skip_training=exp.get("skip_training", False),
            )
            if metrics is not None:
                metrics["seed"] = seed
                per_seed_raw[eid].append(metrics)

        # ── 对当前实验的所有种子进行统计聚合 ──
        if per_seed_raw[eid]:
            agg = aggregate_metrics(per_seed_raw[eid])
            aggregated_results[eid] = agg
            r1_mean = agg.get("rouge1_mean", 0)
            r1_std = agg.get("rouge1_std", 0)
            logging.info(f"📊 [{eid}] 汇总 ({agg['n_seeds']} 种子): "
                          f"ROUGE-1 = {r1_mean:.4f} ± {r1_std:.4f}")

            # 保存聚合结果到 JSON
            agg_path = os.path.join(exp_results_dir, "aggregated.json")
            with open(agg_path, "w", encoding="utf-8") as f:
                json.dump(agg, f, ensure_ascii=False, indent=2)
            logging.info(f"   ✅ 聚合结果已写入: {agg_path}")
        else:
            aggregated_results[eid] = {}
            logging.warning(f"⚠️ [{eid}] 所有种子均失败，无聚合数据")

    # ── 生成全量统计报告 ──
    report_path = generate_ablation_report(
        experiments, aggregated_results, per_seed_raw, RESULTS_DIR
    )

    # ── 总体统计 ──
    total_time = (time.time() - start_time) / 60
    total_success = sum(len(v) for v in per_seed_raw.values())
    total_expected = len(experiments) * len(DEFAULT_SEEDS)

    logging.info(f"\n{'=' * 60}")
    logging.info(f"🎉 多种子消融流水线执行完毕！")
    logging.info(f"   总耗时: {total_time:.2f} 分钟")
    logging.info(f"   成功率: {total_success}/{total_expected}")
    logging.info(f"   统计报告: {report_path}")
    logging.info(f"   日志文件: {log_file}")
    logging.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
