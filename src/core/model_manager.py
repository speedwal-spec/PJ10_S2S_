"""
核心模块：多槽位模型管理器 (Dual-Slot Model Manager)
职责：负责动态扫描硬盘资产、管理双槽位模型热切换、防止显存溢出
"""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import json
import re
import gc
from typing import Dict, Optional, Tuple
from src.configs.config_manager import ExperimentConfig


class ModelManager:
    """
    双槽位模型管理器
    
    特性：
    - 自动扫描硬盘上的模型资产
    - 支持A/B双槽位独立加载
    - 显存安全的热切换机制
    - 防止双倍峰值OOM
    """
    
    def __init__(self, ablation_dir: Optional[str] = None, legacy_dir: Optional[str] = None,
                 config: Optional[ExperimentConfig] = None):
        """
        初始化模型管理器
        
        Args:
            ablation_dir: 消融实验模型目录（优先，默认读配置）
            legacy_dir: 传统模型目录（优先，默认读配置）
            config: 实验配置（可选），用于路径和推理默认值
        """
        # ✅ 获取项目根目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        
        # 从配置读取路径（命令行参数可覆盖）
        if config is not None:
            ablation_dir = ablation_dir or config.paths.ablation_base
            legacy_dir = legacy_dir or config.paths.output_dir
        else:
            # ✅ 使用基于项目根目录的绝对路径
            ablation_dir = ablation_dir or os.path.join(project_root, "checkpoints_ablation")
            legacy_dir = legacy_dir or os.path.join(project_root, "t5-news-checkpoint")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        )
        
        # 从配置加载推理默认值
        if config is not None:
            self.max_source_length = config.data.max_source_length
            self.infer_num_beams = config.inference.num_beams
            self.infer_early_stopping = config.inference.early_stopping
        else:
            self.max_source_length = 512
            self.infer_num_beams = 4
            self.infer_early_stopping = True
        
        # 分离 A/B 两个卡槽的模型与状态
        self.model_a: Optional[AutoModelForSeq2SeqLM] = None
        self.tokenizer_a: Optional[AutoTokenizer] = None
        self.current_model_a: Optional[str] = None
        
        self.model_b: Optional[AutoModelForSeq2SeqLM] = None
        self.tokenizer_b: Optional[AutoTokenizer] = None
        self.current_model_b: Optional[str] = None
        
        # 扫描并注册硬盘上的可用模型资产
        self.registry = self._scan_assets(ablation_dir, legacy_dir)
        print(f"📦 [Model Hub] 共扫描到 {len(self.registry)} 个可用模型资产。")

    def _scan_assets(self, ablation_dir: str, legacy_dir: str) -> Dict[str, dict]:
        """
        扫描目录中的模型资产
        统一以子目录方式扫描两个目录，支持任意数量的实验子目录
        
        Args:
            ablation_dir: 消融实验目录
            legacy_dir: 传统模型目录
            
        Returns:
            模型注册表 {exp_id: profile}
        """
        registry = {}
        
        # 扫描指定目录下的所有子目录（每个子目录为一个实验）
        for scan_dir in [ablation_dir, legacy_dir]:
            if not os.path.exists(scan_dir):
                continue
            for exp_id in os.listdir(scan_dir):
                model_path = os.path.join(scan_dir, exp_id)
                if not os.path.isdir(model_path):
                    continue
                if not os.path.exists(os.path.join(model_path, "config.json")):
                    continue
                # 不覆盖已存在的同名实验（ablation_dir 优先）
                if exp_id not in registry:
                    registry[exp_id] = self._build_model_profile(exp_id, model_path, scan_dir)
        
        # 额外检查：如果 legacy_dir 根目录直接有 config.json，作为 Legacy-Baseline 注册
        if os.path.exists(legacy_dir) and os.path.exists(os.path.join(legacy_dir, "config.json")):
            if "Legacy-Baseline" not in registry:
                registry["Legacy-Baseline"] = self._build_model_profile(
                    "Legacy-Baseline", legacy_dir, legacy_dir
                )
        
        return registry

    def _build_model_profile(self, exp_id: str, model_path: str, scan_dir: str = "") -> dict:
        """
        构建模型档案（包含超参数和评估指标）
        
        Args:
            exp_id: 实验ID
            model_path: 模型路径
            scan_dir: 扫描来源目录（用于查找关联的评测结果文件）
            
        Returns:
            模型档案字典
        """
        profile = {"path": model_path, "hparams": {}, "metrics": {}, "description": ""}
        
        # 1. 读取 train_hparams.json（新版本保存的完整字段，或老版本的部分字段）
        hparams_path = os.path.join(model_path, "train_hparams.json")
        saved_hparams = {}
        if os.path.exists(hparams_path):
            with open(hparams_path, "r", encoding="utf-8") as f:
                saved_hparams = json.load(f)
        profile["hparams"] = dict(saved_hparams)
        
        # 2. 从目录名推断缺失的 hparams（兼容旧 checkpoint）
        inferred = self._infer_hparams_from_name(exp_id)
        for k, v in inferred.items():
            if k not in saved_hparams:
                profile["hparams"][k] = v
        
        # 3. 仍缺失的字段用 default.yaml 兜底
        fallback = self._get_default_hparams()
        for k, v in fallback.items():
            if k not in profile["hparams"]:
                profile["hparams"][k] = v
        
        # 4. 构建描述文案（优先从 ablation config，其次从推断）
        profile["description"] = self._build_description(exp_id, profile["hparams"])
        
        # 5. 读取评估指标（优先从扫描目录找 results_{exp_id}.json）
        metrics_name = f"results_{exp_id}.json"
        for candidate_dir in [scan_dir, "."]:
            if not candidate_dir:
                continue
            metrics_path = os.path.join(candidate_dir, metrics_name)
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        profile["metrics"] = json.load(f)
                except Exception:
                    pass
                break
        
        return profile

    @staticmethod
    def _infer_hparams_from_name(exp_id: str) -> dict:
        """
        从实验目录名推断超参（兼容旧版手动命名）
        
        支持的模式:
          lr_1e3, bs2_ga4, data_120, len_long, no_accum 等
        """
        inferred = {}
        name = exp_id.lower()
        
        # 学习率: lr_1e3 → 0.001, lr_1e4 → 0.0001
        m = re.search(r'lr[_-]?(\d+)e(\d+)', name)
        if m:
            base = float(m.group(1))
            neg_exp = float(m.group(2))
            inferred['lr'] = base * (10 ** -neg_exp)
        m = re.search(r'lr[_-]?(\d+)_?(\d+)', name)  # lr_0_001
        if m and not any(k == 'lr' for k in inferred):
            inferred['lr'] = float(f"{m.group(1)}.{m.group(2)}")
        
        # 批大小: bs2 → 2, bs_8 → 8
        m = re.search(r'bs[_-]?(\d+)', name)
        if m:
            inferred['batch_size'] = int(m.group(1))
        
        # 梯度累积: ga4 → 4, ga_1 → 1
        m = re.search(r'ga[_-]?(\d+)', name)
        if m:
            inferred['grad_accum'] = int(m.group(1))
        if 'no_accum' in name:
            inferred['grad_accum'] = 1
        
        # 训练数据量: data_120 → 120, data_40 → 40
        m = re.search(r'data[_-]?(\d+)', name)
        if m:
            inferred['max_train_samples'] = int(m.group(1))
        
        # 生成长度: len_long → 80, len_short → 20
        if 'len_long' in name:
            inferred['max_target_len'] = 80
        elif 'len_short' in name:
            inferred['max_target_len'] = 20
        
        return inferred

    @staticmethod
    def _get_default_hparams() -> dict:
        """
        从 configs/default.yaml 读取默认超参（作为兜底）
        """
        try:
            from src.configs.config_manager import load_full_config
            _cfg = load_full_config()
            return {
                'lr': _cfg.training.lr,
                'batch_size': _cfg.training.batch_size,
                'grad_accum': _cfg.training.grad_accum,
                'epochs': _cfg.training.epochs,
                'warmup_ratio': _cfg.training.warmup_ratio,
                'seed': _cfg.seed,
            }
        except Exception:
            return {}  # 兜底失败时返回空，Gradio 端会显示 N/A

    @staticmethod
    def _build_description(exp_id: str, hparams: dict) -> str:
        """
        构建模型描述文案
        
        Args:
            exp_id: 实验ID
            hparams: 已补全的超参字典
            
        Returns:
            描述字符串
        """
        if exp_id == "Legacy-Baseline":
            return "T5-small 基础基线（未微调）"
        
        parts = []
        lr = hparams.get('lr', None)
        bs = hparams.get('batch_size', None)
        ga = hparams.get('grad_accum', None)
        data = hparams.get('max_train_samples', None)
        max_len = hparams.get('max_target_len', None)
        
        if lr is not None:
            parts.append(f"lr={lr}")
        if bs is not None:
            parts.append(f"bs={bs}")
        if ga is not None:
            parts.append(f"ga={ga}")
        if data is not None:
            parts.append(f"{data}条数据")
        if max_len is not None:
            parts.append(f"max_len={max_len}")
        
        if parts:
            return " | ".join(parts)
        return exp_id  # 纯 fallback

    def load_model(self, exp_id: str, slot: str = "a") -> str:
        """
        显存安全的双槽热切换：先释放旧模型，再加载新模型
        
        Args:
            exp_id: 实验ID
            slot: 卡槽标识 ('a' 或 'b')
            
        Returns:
            状态消息
        """
        if exp_id not in self.registry: 
            return f"❌ 未找到模型 {exp_id}"
        
        curr_model_id = self.current_model_a if slot == "a" else self.current_model_b
        if exp_id == curr_model_id: 
            return f"✅ 模型已在卡槽 {slot.upper()} 就绪"

        print(f"🔄 正在为卡槽 {slot.upper()} 挂载: {exp_id}...")
        
        # 1. 显式断开旧模型的引用并强制清空显存
        if slot == "a":
            self.model_a, self.tokenizer_a = None, None
        else:
            self.model_b, self.tokenizer_b = None, None
            
        gc.collect()
        if self.device.type == "cuda": 
            torch.cuda.empty_cache()

        # 2. 安全加载新模型
        model_path = self.registry[exp_id]["path"]
        tok = AutoTokenizer.from_pretrained(model_path)
        
        # 关闭低内存模式，手动缝合权重，再安全推入显卡
        net = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=False  # 明确拒绝 accelerate 的虚拟张量加载
        )
        
        # 强行把 T5 分离的 encoder、decoder 和 lm_head 权重绑定到一起
        net.tie_weights() 
        
        # 缝合完毕后，作为一个完整的实体，推入显卡
        net = net.to(self.device)
        net.eval()

        # 3. 绑定到指定卡槽
        if slot == "a":
            self.model_a, self.tokenizer_a, self.current_model_a = net, tok, exp_id
        else:
            self.model_b, self.tokenizer_b, self.current_model_b = net, tok, exp_id
            
        print(f"✅ 卡槽 {slot.upper()} 挂载成功！")
        return f"✅ 成功挂载至卡槽 {slot.upper()}"

    def generate_slot(self, article: str, max_length: int = 40, 
                     length_penalty: float = 0.85, slot: str = "a") -> str:
        """
        指定卡槽进行独立推理
        
        Args:
            article: 输入文章
            max_length: 最大生成长度
            length_penalty: 长度惩罚系数
            slot: 卡槽标识 ('a' 或 'b')
            
        Returns:
            生成的摘要文本
        """
        model = self.model_a if slot == "a" else self.model_b
        tokenizer = self.tokenizer_a if slot == "a" else self.tokenizer_b
        curr_id = self.current_model_a if slot == "a" else self.current_model_b
        
        if not model: 
            return f"⚠️ 卡槽 {slot.upper()} 尚未挂载模型！"
        if not article.strip(): 
            return ""
        
        prefix = self.registry[curr_id]["hparams"].get("prefix", "summarize: ")
        inputs = tokenizer(
            prefix + article, 
            max_length=self.max_source_length, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_length, 
                num_beams=self.infer_num_beams,
                length_penalty=length_penalty, 
                early_stopping=self.infer_early_stopping
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_model_info(self, exp_id: str) -> dict:
        """
        获取模型信息
        
        Args:
            exp_id: 实验ID
            
        Returns:
            模型信息字典
        """
        if exp_id not in self.registry:
            return {}
        return self.registry[exp_id]

    def list_models(self) -> list:
        """
        列出所有可用模型
        
        Returns:
            模型ID列表
        """
        return list(self.registry.keys())
