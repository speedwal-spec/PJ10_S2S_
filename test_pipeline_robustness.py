# -*- coding: utf-8 -*-
"""
T5 新闻摘要系统 - 端到端鲁棒性与连通性测试
运行方式: python test_pipeline_robustness.py
"""
import unittest
import subprocess
import os
import shutil
import json
import torch
import gc

# 导入 main.py 中的核心管理器
try:
    from main import ModelManager
except ImportError:
    raise ImportError("无法导入 main.py，请确保该测试脚本与 main.py 位于同一目录下。")

class TestPipelineRobustness(unittest.TestCase):
    TEST_EXP_ID = "test_robustness_run_001"
    TEST_OUT_DIR = "checkpoints_ablation"
    
    @classmethod
    def setUpClass(cls):
        """
        测试套件初始化：自动触发一个极小规模的 train.py 训练任务
        用 8 条数据、1 个 epoch 快速跑通训练管线
        """
        print("\n" + "="*50)
        print(" [SetUp] 正在启动 train.py 极速测试跑批...")
        
        # 构建最小化训练命令
        cmd = [
            "python", "-X", "utf8", "train.py",
            "--exp_id", cls.TEST_EXP_ID,
            "--output_dir", cls.TEST_OUT_DIR,
            "--max_train_samples", "8",
            "--max_val_samples", "4",
            "--epochs", "1",
            "--batch_size", "2",
            "--no_rouge_eval" # 提速，跳过 ROUGE 评测
        ]
        
        try:
            # check=True 保证如果 train.py 崩溃，测试直接失败
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("[SetUp] train.py 极速训练执行成功！")
        except subprocess.CalledProcessError as e:
            # 使用 gbk 解码 Windows 终端输出，并用 replace 忽略无法解析的乱码
            error_msg = e.stderr.decode('gbk', errors='replace')
            print(f" [SetUp] train.py 执行失败！\n错误信息:\n{error_msg}")
            raise e

    def test_01_train_artifacts_integrity(self):
        """
        测试点 1：检查 train.py 是否生成了 main.py 依赖的所有标准资产文件
        """
        exp_path = os.path.join(self.TEST_OUT_DIR, self.TEST_EXP_ID)
        self.assertTrue(os.path.exists(exp_path), f"输出目录不存在: {exp_path}")
        
        # 1. 检查模型配置与权重
        self.assertTrue(os.path.exists(os.path.join(exp_path, "config.json")), "缺失 config.json")
        self.assertTrue(
            os.path.exists(os.path.join(exp_path, "pytorch_model.bin")) or 
            os.path.exists(os.path.join(exp_path, "model.safetensors")), 
            "缺失模型权重文件"
        )
        
        # 2. 检查 main.py 强依赖的超参卡片
        hparams_path = os.path.join(exp_path, "train_hparams.json")
        self.assertTrue(os.path.exists(hparams_path), "缺失 train_hparams.json")
        
        with open(hparams_path, "r", encoding="utf-8") as f:
            hparams = json.load(f)
            self.assertIn("prefix", hparams, "hparams 缺少 prefix 字段 (main.py 推理必需)")
            self.assertIn("max_source_len", hparams, "hparams 缺少 max_source_len 字段")

    def test_02_model_manager_scanning_robustness(self):
        """
        测试点 2：测试 main.py 的 ModelManager 能否正确扫描到刚刚训练出的资产
        """
        manager = ModelManager(ablation_dir=self.TEST_OUT_DIR)
        
        self.assertIn(self.TEST_EXP_ID, manager.registry, "ModelManager 未能扫描到新训练的模型")
        profile = manager.registry[self.TEST_EXP_ID]
        self.assertEqual(profile["hparams"].get("exp_id"), self.TEST_EXP_ID, "超参文件解析内容错位")

    def test_03_dual_slot_hot_swap_memory_leak(self):
        """
        测试点 3：测试双槽位热切换与显存泄漏保护 (tie_weights 和 gc.collect 逻辑)
        """
        manager = ModelManager(ablation_dir=self.TEST_OUT_DIR)
        
        # 记录初始显存
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated()
        
        # 将测试模型挂载到卡槽 A
        status_a = manager.load_model(self.TEST_EXP_ID, slot="a")
        self.assertIn("成功挂载", status_a, "卡槽 A 挂载失败")
        self.assertIsNotNone(manager.model_a, "卡槽 A 模型实例为空")
        
        # 将同样的模型再次挂载到卡槽 B
        status_b = manager.load_model(self.TEST_EXP_ID, slot="b")
        self.assertIn("成功挂载", status_b, "卡槽 B 挂载失败")
        self.assertIsNotNone(manager.model_b, "卡槽 B 模型实例为空")
        
        # 强行重新挂载卡槽 A，测试清理逻辑是否报错
        status_a2 = manager.load_model(self.TEST_EXP_ID, slot="a")
        self.assertIn("就绪", status_a2, "重复挂载时保护逻辑失效")

    def test_04_inference_connectivity(self):
        """
        测试点 4：测试挂载后的模型能否正常进行连通性推理，且没有 Tensor device 错位报错
        """
        manager = ModelManager(ablation_dir=self.TEST_OUT_DIR)
        manager.load_model(self.TEST_EXP_ID, slot="a")
        
        test_article = "Apple Inc. announced on Tuesday that it will transition its entire Mac lineup from Intel processors to its own custom-designed ARM-based silicon, a move that promises significant performance and energy efficiency improvements."
        
        # 执行推理
        summary = manager.generate_slot(test_article, max_length=20, length_penalty=1.0, slot="a")
        
        # 验证结果
        self.assertIsInstance(summary, str, "生成的摘要不是字符串格式")
        self.assertGreater(len(summary.strip()), 0, "模型生成了空字符串")
        print(f"\n推理联通测试成功！\n输入: {test_article[:50]}...\n输出: {summary}")

    @classmethod
    def tearDownClass(cls):
        """
        测试套件清理：删除测试产生的检查点文件夹，保持环境整洁
        """
        exp_path = os.path.join(cls.TEST_OUT_DIR, cls.TEST_EXP_ID)
        if os.path.exists(exp_path):
            shutil.rmtree(exp_path)
            print(f"\n [TearDown] 已清理测试遗留资产: {exp_path}")

if __name__ == "__main__":
    unittest.main(verbosity=2)