"""
只运行4组新实验的专用脚本
"""
import os
import subprocess
import time

NEW_EXPERIMENTS = [
    "full_data_2000.yaml",
    "lr_5e_4.yaml",
    "optimal_combo.yaml",
    "epochs_10.yaml",
]

OUTPUT_BASE = "checkpoints_ablation"
EVAL_MAX_SAMPLES = 1000

print("=" * 60)
print("🚀 运行4组新实验\n")
start_time = time.time()

for cfg_file in NEW_EXPERIMENTS:
    cfg_name = cfg_file.replace(".yaml", "")
    result_json = f"results_{cfg_name}.json"

    print(f"\n{'=' * 60}")
    print(f"▶ 训练: {cfg_name}")
    print(f"{'=' * 60}")

    # 训练
    train_cmd = (
        f"python train.py "
        f"--config configs/ablation/{cfg_file} "
        f"--output_dir {OUTPUT_BASE}"
    )
    result = subprocess.run(train_cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ {cfg_name} 训练失败，跳过评测")
        continue

    # 评测
    print(f"\n📊 评测: {cfg_name}")
    ckpt_path = os.path.join(OUTPUT_BASE, cfg_name)
    eval_cmd = (
        f"python evaluate_rouge.py "
        f"--ckpt {ckpt_path} "
        f"--output_json {result_json} "
        f"--max_samples {EVAL_MAX_SAMPLES}"
    )
    subprocess.run(eval_cmd, shell=True)
    print(f"✅ {cfg_name} 完成，结果已保存到 {result_json}")

total = (time.time() - start_time) / 60
print(f"\n{'=' * 60}")
print(f"🎉 4组新实验全部完成！总耗时: {total:.1f} 分钟")
print(f"   结果文件: results_full_data_2000.json, results_lr_5e4.json, "
      f"results_lr_0.001_data_120.json, results_epochs_10.json")
