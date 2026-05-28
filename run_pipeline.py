import os
import subprocess
import time
import logging

# ==========================================
# 0. 工业级全局日志配置 (Logging Setup)
# ==========================================
# 自动生成带时间戳的日志文件，例如：pipeline_20260528_1200.log
log_filename = f"pipeline_{time.strftime('%Y%m%d_%H%M')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'), # 永久保存到硬盘防丢失
        logging.StreamHandler() # 同时显示在屏幕上
    ]
)

def run_command(cmd, desc):
    """带日志记录的命令行执行器"""
    logging.info(f"\n{'='*60}")
    logging.info(f"⚙️  [Pipeline 阶段]: {desc}")
    logging.info(f"💻 [执行命令]: {cmd}")
    logging.info(f"{'='*60}\n")
    # check=True 表示如果该命令报错（如 OOM），会向外抛出异常供上层捕获
    subprocess.run(cmd, shell=True, check=True)

def main():
    logging.info("🚀 T5-News 自动化多维消融实验流水线启动！\n")
    start_time = time.time()

    # 1. 确保数据就绪
    try:
        run_command("python prepare_data.py", "初始化数据管道")
    except Exception as e:
        logging.error(f"❌ 数据准备失败，流水线中止: {e}")
        return

    # ==========================================
    # 2. 定义严谨的消融实验矩阵 (完美融合 V1 的科研严谨性)
    # ==========================================
    ablation_matrix = [
        # 1. 对照组 (Baseline)
        {"id": "Exp-01_Baseline",  "lr": 3e-4, "bs": 4, "ga": 2, "samples": 2000, "max_src": 512, "max_tgt": 40, "epochs": 3},
        
        # 2. 学习率消融 (Learning Rate)
        {"id": "Exp-02_LowLR",     "lr": 1e-4, "bs": 4, "ga": 2, "samples": 2000, "max_src": 512, "max_tgt": 40, "epochs": 3},
        {"id": "Exp-03_HighLR",    "lr": 1e-3, "bs": 4, "ga": 2, "samples": 2000, "max_src": 512, "max_tgt": 40, "epochs": 3},
        
        # 3. 显存与批次消融 (Batch Size & Grad Accum)
        {"id": "Exp-04_SmallBS",   "lr": 3e-4, "bs": 2, "ga": 4, "samples": 2000, "max_src": 512, "max_tgt": 40, "epochs": 3},
        {"id": "Exp-05_LargeBS",   "lr": 3e-4, "bs": 8, "ga": 1, "samples": 2000, "max_src": 512, "max_tgt": 40, "epochs": 3},
        {"id": "Exp-06_NoAccum",   "lr": 3e-4, "bs": 4, "ga": 1, "samples": 2000, "max_src": 512, "max_tgt": 40, "epochs": 3},
        
        # 4. 序列长度消融 (Sequence Length)
        {"id": "Exp-07_ShortSeq",  "lr": 3e-4, "bs": 4, "ga": 2, "samples": 2000, "max_src": 256, "max_tgt": 30, "epochs": 3},
        {"id": "Exp-08_LongSeq",   "lr": 3e-4, "bs": 4, "ga": 2, "samples": 2000, "max_src": 768, "max_tgt": 50, "epochs": 3},
        
        # 5. 数据规模 Scaling Law (Data Volume)
        {"id": "Exp-09_Data40",    "lr": 3e-4, "bs": 4, "ga": 2, "samples": 40,   "max_src": 512, "max_tgt": 40, "epochs": 3},
        {"id": "Exp-10_Data120",   "lr": 3e-4, "bs": 4, "ga": 2, "samples": 120,  "max_src": 512, "max_tgt": 40, "epochs": 3},
    ]

    # 所有模型统一存放在这个大目录下，里面再按 exp_id 严格隔离
    OUTPUT_BASE = "checkpoints_ablation"
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    logging.info(f"📊 本次流水线共包含 {len(ablation_matrix)} 组独立消融实验。\n")

    for exp in ablation_matrix:
        logging.info(f"\n" + "🌟"*40)
        logging.info(f"正在启动子任务 >>> {exp['id']}")
        logging.info("🌟"*40)
        
        # 构建训练命令，补齐了 --max_target_len 参数！
        train_cmd = (
            f"python train.py "
            f"--exp_id {exp['id']} "
            f"--output_dir {OUTPUT_BASE} "
            f"--lr {exp['lr']} "
            f"--batch_size {exp['bs']} "
            f"--grad_accum {exp['ga']} "
            f"--epochs {exp['epochs']} "
            f"--max_train_samples {exp['samples']} "
            f"--max_source_len {exp['max_src']} "
            f"--max_target_len {exp['max_tgt']} "  
            f"--no_rouge_eval" 
        )
        
        # 组装评估命令
        ckpt_path = os.path.join(OUTPUT_BASE, exp['id'])
        result_json = f"results_{exp['id']}.json"
        
        eval_cmd = (
            f"python evaluate_rouge.py "
            f"--ckpt {ckpt_path} "
            f"--output_json {result_json} "
            f"--max_samples 1000" 
        )

        # ==========================================
        # 核心升级：OOM 防御与高可用容错机制
        # ==========================================
        try:
            run_command(train_cmd, f"正式炼丹点火 - {exp['id']}")
            run_command(eval_cmd, f"客观指标评测 - {exp['id']}")
            logging.info(f"✅ {exp['id']} 顺利完结，指标已固化至 {result_json}")

        except subprocess.CalledProcessError as e:
            logging.warning(f"❌ 警告: {exp['id']} 运行时发生致命错误 (大概率是显存溢出 OOM)。")
            logging.info("⏭️ 流水线具有高容错性，正在自动切入下一组消融实验...")
            continue

    # 3. 总体耗时统计与收尾指引
    total_time = (time.time() - start_time) / 60
    logging.info(f"\n🎉 满血版流水线执行完毕！共耗时: {total_time:.2f} 分钟。")
    logging.info("   1. 运行 `tensorboard --logdir=runs` -> 在浏览器查看各路神仙打架的 Loss 曲线。")
    logging.info("   2. 运行 `python generate_comparison_plots.py` -> 自动扫描刚才生成的 json 文件，一键出精美柱状图。")

if __name__ == "__main__":
    main()