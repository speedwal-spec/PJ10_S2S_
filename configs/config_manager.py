# configs/config_manager.py
"""
配置管理器：
1. 支持YAML配置文件读取
2. 支持配置继承（extends关键字）
3. 支持命令行覆盖
4. 支持类型安全和自动补全（dataclass）
"""
import os
import yaml
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


# ==========================================
# 1. 数据类定义（提供IDE自动补全）
# ==========================================

@dataclass
class DataConfig:
    manifest: str = "data_manifest.json"
    max_train_samples: int = 2000
    max_val_samples: int = 400
    max_source_length: int = 512
    max_target_length: int = 40
    prefix: str = "summarize: "
    min_val_for_rouge: int = 128


@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 4
    grad_accum: int = 2
    lr: float = 0.0003
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    label_pad_token_id: int = -100


@dataclass
class EarlyStoppingConfig:
    patience: int = 3
    monitor: str = "val_loss"


@dataclass
class LoggingConfig:
    tensorboard_dir: str = "runs"


@dataclass
class InferenceConfig:
    max_new_tokens: int = 40
    num_beams: int = 4
    length_penalty: float = 0.85
    no_repeat_ngram: int = 2
    early_stopping: bool = True


@dataclass
class EvaluationConfig:
    split: str = "validation"
    max_samples: int = 1000
    batch_size: int = 4
    output_json: str = ""


@dataclass
class VisualizationConfig:
    output_dir: str = "comparison_plots"
    font: tuple = ("SimHei", "Microsoft YaHei", "DejaVu Sans")
    dpi: int = 150


@dataclass
class PathConfig:
    output_dir: str = "t5-news-checkpoint"
    ablation_base: str = "checkpoints_ablation"
    cache_dir: str = "data_cache"
    samples_file: str = "sample_articles_20.json"
    results_dir: str = ""
    tensorboard_dir: str = "runs"
    visualization_dir: str = "comparison_plots"


@dataclass
class EnvironmentConfig:
    hf_endpoint: str = "https://hf-mirror.com"
    device: str = "auto"
    amp: str = "auto"
    cpu_threads: int = 0
    pin_memory: bool = False
    dataloader_workers: int = 0


@dataclass
class ExperimentConfig:
    """顶级配置 - 包含所有子配置"""
    id: str = "debug_run"
    description: str = ""
    seed: int = 42
    model_name: str = "google-t5/t5-small"
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)


# ==========================================
# 2. YAML加载器（支持继承）
# ==========================================

def _resolve_extends(raw: dict, base_dir: str) -> dict:
    """处理 extends 关键字，递归合并配置"""
    extends = raw.pop("extends", None)
    if extends is None:
        return raw

    # 解析继承链
    parent_path = os.path.join(base_dir, extends)
    parent_path = os.path.normpath(parent_path)

    with open(parent_path, "r", encoding="utf-8") as f:
        parent_raw = yaml.safe_load(f)

    # 递归解析父配置的 extends
    parent_dir = os.path.dirname(parent_path)
    parent_raw = _resolve_extends(parent_raw, parent_dir)

    # 合并（子配置覆盖父配置）
    merged = {**parent_raw, **raw}
    return merged


def load_config(yaml_path: str) -> ExperimentConfig:
    """
    从YAML文件加载配置

    用法：
        config = load_config("configs/ablation/baseline.yaml")
        print(config.training.lr)      # 0.0003
        print(config.data.prefix)      # "summarize: "
    """
    yaml_path = os.path.normpath(yaml_path)
    base_dir = os.path.dirname(yaml_path)

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # 处理继承
    raw = _resolve_extends(raw, base_dir)

    # 构造数据类
    config = ExperimentConfig()
    config.id = raw.get("experiment", {}).get("id", "debug_run")
    config.description = raw.get("experiment", {}).get("description", "")
    config.seed = raw.get("project", {}).get("seed", 42)
    config.model_name = raw.get("model", {}).get("name", "google-t5/t5-small")

    # 子配置
    data_raw = raw.get("data", {})
    config.data = DataConfig(**data_raw)

    train_raw = raw.get("training", {})
    config.training = TrainingConfig(**train_raw)

    es_raw = raw.get("early_stopping", {})
    config.early_stopping = EarlyStoppingConfig(**es_raw)

    log_raw = raw.get("logging", {})
    config.logging = LoggingConfig(**log_raw)

    infer_raw = raw.get("inference", {})
    config.inference = InferenceConfig(**infer_raw)

    eval_raw = raw.get("evaluation", {})
    config.evaluation = EvaluationConfig(**eval_raw)

    viz_raw = raw.get("visualization", {})
    config.visualization = VisualizationConfig(**viz_raw)

    path_raw = raw.get("paths", {})
    config.paths = PathConfig(**path_raw)

    env_raw = raw.get("environment", {})
    config.environment = EnvironmentConfig(**env_raw)

    return config


def load_full_config(
    default_path: str = "configs/default.yaml",
    paths_path: str = "configs/paths.yaml",
    hardware_path: str = "configs/hardware.yaml",
) -> ExperimentConfig:
    """
    加载完整配置：default.yaml + paths.yaml + hardware.yaml 自动合并。

    优先级：hardware.yaml > paths.yaml > default.yaml
    """
    base_dir = os.path.dirname(os.path.normpath(default_path))

    # 1. 加载 default.yaml
    config = load_config(default_path)

    # 2. 合并 paths.yaml（如果存在）
    paths_full = os.path.normpath(paths_path) if not os.path.isabs(paths_path) else paths_path
    if os.path.isfile(paths_full):
        with open(paths_full, "r", encoding="utf-8") as f:
            paths_raw = yaml.safe_load(f) or {}
        path_raw = paths_raw.get("paths", {})
        for k, v in path_raw.items():
            if v is not None:
                setattr(config.paths, k, v)
        env_raw = paths_raw.get("environment", {})
        for k, v in env_raw.items():
            if v is not None:
                setattr(config.environment, k, v)

    # 3. 合并 hardware.yaml（如果存在，覆盖 env 字段）
    hw_full = os.path.normpath(hardware_path) if not os.path.isabs(hardware_path) else hardware_path
    if os.path.isfile(hw_full):
        with open(hw_full, "r", encoding="utf-8") as f:
            hw_raw = yaml.safe_load(f) or {}
        dev_raw = hw_raw.get("device", {})
        if dev_raw.get("preferred"):
            config.environment.device = dev_raw["preferred"]
        if dev_raw.get("cpu_threads"):
            config.environment.cpu_threads = dev_raw["cpu_threads"]
        mem_raw = hw_raw.get("memory", {})
        if mem_raw.get("amp") is not None:
            config.environment.amp = "fp16" if mem_raw["amp"] else "off"
        if mem_raw.get("pin_memory") is not None:
            config.environment.pin_memory = mem_raw["pin_memory"]
        if mem_raw.get("dataloader_workers") is not None:
            config.environment.dataloader_workers = mem_raw["dataloader_workers"]

    return config


# ==========================================
# 3. 命令行参数 + 配置文件 混合解析
# ==========================================

def merge_with_cli(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """
    命令行参数覆盖配置文件的值
    （如果命令行指定了，就用命令行的）
    """
    # 常用的覆盖项
    overrides = {
        'lr': ('training', 'lr'),
        'batch_size': ('training', 'batch_size'),
        'grad_accum': ('training', 'grad_accum'),
        'epochs': ('training', 'epochs'),
        'patience': ('early_stopping', 'patience'),
        'max_train_samples': ('data', 'max_train_samples'),
        'max_val_samples': ('data', 'max_val_samples'),
        'max_source_len': ('data', 'max_source_length'),
        'max_target_len': ('data', 'max_target_length'),
    }

    for cli_key, (section, attr) in overrides.items():
        cli_val = getattr(args, cli_key, None)
        if cli_val is not None:
            section_obj = getattr(config, section)
            setattr(section_obj, attr, cli_val)

    return config
