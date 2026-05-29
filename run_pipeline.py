import os
import subprocess
import time
import logging
import glob
from configs.config_manager import load_config, load_full_config

# ==========================================
# 0. 工业级全局日志配置 (Logging Setup)
# ==========================================
log_filename = f"pipeline_{time.strftime('%Y%m%d_%H%M')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# ==========================================
# 1. 从 YAML 配置文件自动读取消融实验矩阵
# ==========================================
def load_ablation_experiments(config_dir: str = "configs/ablation") -> list:
    """
    扫描 configs/ablation/ 目录下的所有YAML文件，自动构建实验矩阵。
    
    Returns:
        list of dict，每个dict包含该实验的配置参数
    """
    yaml_files = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))
    
    if not yaml_files:
        logging.warning(f"⚠️ 未在 {config_dir}/ 找到任何YAML配置文件！")
        return []
    
    experiments = []
    for yaml_path in yaml_files:
        # 跳过 baseline.yaml（作为继承父类，不会直接运行）
        if "baseline" in os.path.basename(yaml_path).lower():
            continue
            
        try:
            config = load_config(yaml_path)
            
            exp = {
                "id": config.id,
                "lr": config.training.lr,
                "bs": config.training.batch_size,
                "ga": config.training.grad_accum,
                "epochs": config.training.epochs,
                "samples": config.data.max_train_samples,
                "max_src": config.data.max_source_length,
                "max_tgt": config.data.max_target_length,
                "prefix": config.data.prefix,
                "config_file": os.path.basename(yaml_path),
            }
            experiments.append(exp)
            logging.info(f"📋 发现实验 [{exp['id']}] <- {exp['config_file']}")
            
        except Exception as e:
            logging.error(f"❌ 加载 {yaml_path} 失败: {e}")
            continue
    
    return experiments


def run_command(cmd, desc):
    """带日志记录的命令行执行器"""
    logging.info(f"\n{'='*60}")
    logging.info(f"⚙️  [Pipeline 阶段]: {desc}")
    logging.info(f"💻 [执行命令]: {cmd}")
    logging.info(f"{'='*60}\n")
    subprocess.run(cmd, shell=True, check=True)


def _find_ckpt(exp_id: str, *search_dirs: str):
    """在多个目录中查找实验的 checkpoint"""
    for base_dir in search_dirs:
        candidate = os.path.join(base_dir, exp_id)
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate
    return None


def main():
    logging.info("🚀 T5-News 自动化多维消融实验流水线启动！\n")
    start_time = time.time()
    
    # 从 default.yaml 加载路径和评测配置
    cfg = load_full_config()
    OUTPUT_BASE = cfg.paths.ablation_base
    EVAL_MAX_SAMPLES = cfg.evaluation.max_samples
    LEGACY_DIR = "t5-news-checkpoint"
    logging.info(f"📋 消融输出目录: {OUTPUT_BASE} | 兼容目录: {LEGACY_DIR} | 评测样本数: {EVAL_MAX_SAMPLES}")
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # 1. 确保数据就绪
    try:
        run_command("python prepare_data.py", "初始化数据管道")
    except Exception as e:
        logging.error(f"❌ 数据准备失败，流水线中止: {e}")
        return

    # ==========================================
    # 2. 从 YAML 配置文件自动加载实验矩阵
    # ==========================================
    ablation_matrix = load_ablation_experiments()
    
    if not ablation_matrix:
        logging.error("❌ 未找到任何实验配置，流水线中止。请先在 configs/ablation/ 下添加YAML文件")
        return

    logging.info(f"📊 本次流水线共包含 {len(ablation_matrix)} 组消融实验（来自 configs/ablation/）\n")

    for exp in ablation_matrix:
        logging.info(f"\n" + "🌟"*40)
        logging.info(f"正在启动子任务 >>> {exp['id']} (来自 {exp['config_file']})")
        logging.info("🌟"*40)

        # 构建训练命令
        train_cmd = (
            f"python train.py "
            f"--config configs/ablation/{exp['config_file']} "
            f"--output_dir {OUTPUT_BASE} "
        )

        # 先搜索已有 checkpoint（兼容 checkpoints_ablation/ 和 t5-news-checkpoint/）
        ckpt_path = _find_ckpt(exp['id'], OUTPUT_BASE, LEGACY_DIR)

        # 如果没有现成的 checkpoint，先训练
        if ckpt_path is None:
            try:
                run_command(train_cmd, f"训练 - {exp['id']}")
            except subprocess.CalledProcessError:
                logging.warning(f"❌ {exp['id']} 训练失败，跳过")
                continue
            # 训练后重新定位 checkpoint
            ckpt_path = _find_ckpt(exp['id'], OUTPUT_BASE, LEGACY_DIR)
            if ckpt_path is None:
                logging.warning(f"❌ {exp['id']} 训练后仍未找到 checkpoint，跳过")
                continue
            logging.info(f"📁 训练完成，在 {os.path.dirname(ckpt_path)}/ 找到 checkpoint")
        else:
            logging.info(f"📁 已有现成 checkpoint（{ckpt_path}），跳过训练")

        # 评测
        result_json = f"results_{exp['id']}.json"
        eval_cmd = (
            f"python evaluate_rouge.py "
            f"--ckpt {ckpt_path} "
            f"--output_json {result_json} "
            f"--max_samples {EVAL_MAX_SAMPLES}"
        )

        try:
            run_command(eval_cmd, f"评测 - {exp['id']}")
            logging.info(f"✅ {exp['id']} 顺利完结，指标已固化至 {result_json}")
        except subprocess.CalledProcessError:
            logging.warning(f"❌ {exp['id']} 评测失败，跳过")
            continue

    # 3. 总体耗时统计与收尾指引
    total_time = (time.time() - start_time) / 60
    logging.info(f"\n🎉 满血版流水线执行完毕！共耗时: {total_time:.2f} 分钟。")
    logging.info("   1. 运行 `tensorboard --logdir=runs` -> 在浏览器查看各路神仙打架的 Loss 曲线。")
    logging.info("   2. 运行 `python generate_comparison_plots.py` -> 自动扫描刚才生成的 json 文件，一键出精美柱状图。")

if __name__ == "__main__":
    main()
