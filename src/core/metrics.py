"""
统一评测指标模块
==========================
提供 ROUGE / BERTScore / LLM-as-a-Judge 的统一接口，
所有指标均以 Dict[str, float] 返回
"""
import os
import sys
import json
import re
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. ROUGE

def compute_rouge(
    preds: List[str],
    refs: List[str],
) -> Dict[str, float]:
    """
    计算 ROUGE-1 / ROUGE-2 / ROUGE-L（F1 均值）

    Args:
        preds: 模型生成的文本列表
        refs: 参考文本列表

    Returns:
        {"rouge1": float, "rouge2": float, "rougeL": float, "n": int}
        依赖缺失时返回空 dict
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("[Metrics] rouge-score 未安装，跳过 ROUGE。pip install rouge-score",
              file=sys.stderr)
        return {}

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    r1, r2, rl = [], [], []

    for p, g in zip(preds, refs):
        p = (p or "").strip()
        g = (g or "").strip()
        if not g:
            continue
        sc = scorer.score(g, p)
        r1.append(sc["rouge1"].fmeasure)
        r2.append(sc["rouge2"].fmeasure)
        rl.append(sc["rougeL"].fmeasure)

    n = len(r1)
    if n == 0:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "n": 0}

    return {
        "rouge1": float(np.mean(r1)),
        "rouge2": float(np.mean(r2)),
        "rougeL": float(np.mean(rl)),
        "n": n,
    }


# 2. BERTScore（稠密向量语义相似度）

_BERTSCORE_DEFAULT_MODEL = "distilbert/distilbert-base-uncased"
# 备选模型（按体积从小到大）
_BERTSCORE_FALLBACK_MODELS = [
    "roberta-base",           # ~500MB
    "roberta-large",           # ~1.5GB
    "microsoft/deberta-xlarge-mnli",  # ~1.5GB
]

def compute_bertscore(
    preds: List[str],
    refs: List[str],
    model_type: Optional[str] = None,
    lang: str = "en",
    device: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    计算 BERTScore（Precision / Recall / F1）
    Args:
        preds: 生成文本列表
        refs: 参考文本列表
        model_type: BERT 模型名称，默认使用 distilbert/distilbert-base-uncased（~270MB）

        lang: 语言代码，默认 "en"
        device: 计算设备，默认自动检测
        verbose: 是否显示 BERTScore 内部日志

    Returns:
        {"bertscore_p": float, "bertscore_r": float, "bertscore_f1": float, "n": int}
        依赖缺失或模型下载失败时返回空 dict
    """
    try:
        from bert_score import score as bertscore_score
    except ImportError:
        print("[Metrics] bert-score 未安装，跳过 BERTScore。pip install bert-score",
              file=sys.stderr)
        return {}

    if device is None:
        device = "cuda:0" if _has_cuda() else "cpu"

    if not os.environ.get("HF_ENDPOINT"):
        try:
            import yaml
            paths_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs", "paths.yaml"
            )
            if os.path.isfile(paths_path):
                with open(paths_path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
                env_cfg = raw.get("environment", {})
                hf_endpoint = env_cfg.get("hf_endpoint", "")
                if hf_endpoint:
                    os.environ["HF_ENDPOINT"] = hf_endpoint
                    os.environ["HUGGINGFACE_HUB_ENDPOINT"] = hf_endpoint
                    print(f"  [Metrics] 自动设置 HF_ENDPOINT={hf_endpoint}（来自 paths.yaml）")
        except Exception:
            pass  # 静默失败，不影响后续

    # 构建待尝试的模型列表（先试首选，再依次试备选）
    primary_model = model_type or _BERTSCORE_DEFAULT_MODEL
    models_to_try = [primary_model] + [
        m for m in _BERTSCORE_FALLBACK_MODELS if m != primary_model
    ]

    for model in models_to_try:
        try:
            P, R, F1 = bertscore_score(
                preds, refs,
                lang=lang,
                model_type=model,
                device=device,
                verbose=verbose,
            )
            if model != primary_model:
                print(f"  [Metrics] BERTScore 使用备选模型 {model}（{primary_model} 不可用）")
            return {
                "bertscore_p": float(P.mean().item()),
                "bertscore_r": float(R.mean().item()),
                "bertscore_f1": float(F1.mean().item()),
                "n": len(preds),
            }
        except Exception as e:
            print(f"  [Metrics] BERTScore ({model}) 加载失败: {e}", file=sys.stderr)
            continue

    print("=" * 60, file=sys.stderr)
    print("[Metrics] BERTScore 所有备选模型均加载失败。", file=sys.stderr)
    print("  可能的原因和解决方案:", file=sys.stderr)
    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if not hf_endpoint:
        print("  1) 设置 HuggingFace 镜像（国内用户必需）:", file=sys.stderr)
        print("     $env:HF_ENDPOINT = 'https://hf-mirror.com'", file=sys.stderr)
    else:
        print(f"  当前 HF_ENDPOINT={hf_endpoint}，如仍失败请检查网络或尝试其他镜像", file=sys.stderr)
    print("  2) 检查网络连接，确认可以访问 huggingface.co", file=sys.stderr)
    print("  3) 手动安装轻量模型缓存:", file=sys.stderr)
    print("     python -c \"from bert_score import score; score(['test'], ['test'], model_type='roberta-base')\"", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    return {}


# 3. LLM-as-a-Judge（大模型评分）

_LLM_JUDGE_SYSTEM_PROMPT = """你是一个专业的新闻标题质量评估助手。请对以下模型生成的标题进行评分。

评分维度（每项 1-5 分）：
1. 事实一致性 (factual_consistency) —— 标题信息是否与原文一致，无幻觉
2. 凝练度 (conciseness) —— 标题是否足够精炼，无冗余词
3. 语义对齐 (semantic_alignment) —— 标题是否抓住新闻核心要点
4. 吸引力 (attractiveness) —— 标题是否简洁有力、抓人眼球

请以 JSON 格式返回评分，不要额外解释。示例：
{"factual_consistency": 4, "conciseness": 5, "semantic_alignment": 4, "attractiveness": 4}
"""


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _call_llm_api(
    prompt: str,
    api_key: str,
    api_base: str,
    model: str,
    max_retries: int = 2,
) -> Optional[str]:
    """调用 OpenAI 兼容的 LLM API"""
    try:
        import requests
    except ImportError:
        print("  [Metrics] requests 未安装，无法调用 LLM API", file=sys.stderr)
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _LLM_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 256,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{api_base.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                return content
            else:
                print(f"  [Metrics] LLM API 返回 {resp.status_code}: {resp.text[:200]}",
                      file=sys.stderr)
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            print(f"  [Metrics] LLM API 调用失败: {e}", file=sys.stderr)
    return None


def _parse_llm_score(llm_response: str) -> Optional[Dict[str, float]]:
    """从 LLM 响应中解析 JSON 评分"""
    if not llm_response:
        return None

    # 尝试提取 JSON
    json_match = re.search(r'\{[^}]+\}', llm_response, re.DOTALL)
    if json_match:
        try:
            scores = json.loads(json_match.group())
            dims = ["factual_consistency", "conciseness", "semantic_alignment", "attractiveness"]
            result = {}
            for d in dims:
                v = scores.get(d)
                if v is not None:
                    result[f"llm_{d}"] = float(v)
            if result:
                result["llm_avg"] = float(np.mean(list(result.values())))
                return result
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def compute_llm_judge(
    preds: List[str],
    refs: List[str],
    articles: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model: str = "gpt-4o-mini",
    max_samples: int = 20,
    max_workers: int = 5,
) -> Dict[str, float]:
    """
    LLM-as-a-Judge：使用大模型对生成标题进行多维评分

    评分维度：factual_consistency, conciseness, semantic_alignment, attractiveness

    注意：
    - 需要 OpenAI 兼容的 API（可通过环境变量 LLM_API_KEY / LLM_API_BASE 配置）
    - 默认只评测前 20 条（避免 API 费用过高）
    - 缺失依赖时返回空 dict
    Args:
        preds: 生成文本列表
        refs: 参考文本列表
        articles: 原始新闻文本列表（用于事实一致性判断），不传时用 refs 代替
        api_key: LLM API Key，默认从环境变量 LLM_API_KEY 读取
        api_base: LLM API 地址，默认从环境变量 LLM_API_BASE 读取
        model: 模型名，默认 gpt-4o-mini
        max_samples: 最多评测条数
        max_workers: 并发请求数

    Returns:
        {"llm_factual_consistency": float, "llm_conciseness": float, ...,
         "llm_avg": float, "n": int}
    """
    api_key = api_key or os.environ.get("LLM_API_KEY")
    api_base = api_base or os.environ.get("LLM_API_BASE")

    if not api_key or not api_base:
        print("  [Metrics] LLM-as-a-Judge 未配置：需要设置 LLM_API_KEY 和 LLM_API_BASE 环境变量",
              file=sys.stderr)
        return {}

    # 限制样本数
    n = min(len(preds), max_samples)
    preds_sample = preds[:n]
    refs_sample = refs[:n]
    articles_sample = (articles[:n] if articles else refs_sample)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, (pred, ref, article) in enumerate(zip(preds_sample, refs_sample, articles_sample)):
            prompt = (
                f"## 原文\n{article[:800]}\n\n"
                f"## 参考标题\n{ref}\n\n"
                f"## 模型生成的标题\n{pred}\n\n"
                f"请评估上述模型生成的标题质量。"
            )
            future = executor.submit(
                _call_llm_api, prompt, api_key, api_base, model
            )
            futures[future] = i

        for future in as_completed(futures):
            response = future.result()
            parsed = _parse_llm_score(response)
            if parsed:
                results.append(parsed)

    if not results:
        return {}

    # 聚合
    aggregated = {}
    dims = list(results[0].keys())
    for d in dims:
        values = [r[d] for r in results if d in r]
        if values:
            aggregated[d] = float(np.mean(values))
    aggregated["n"] = len(results)
    return aggregated


# 4. 统一入口

def compute_all_metrics(
    preds: List[str],
    refs: List[str],
    articles: Optional[List[str]] = None,
    enable_rouge: bool = True,
    enable_bertscore: bool = False,
    enable_llm_judge: bool = False,
    llm_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Args:
        preds: 生成文本列表
        refs: 参考文本列表
        articles: 原始新闻文本（LLM 事实一致性需要）
        enable_rouge: 是否计算 ROUGE
        enable_bertscore: 是否计算 BERTScore
        enable_llm_judge: 是否使用 LLM 评分
        llm_config: LLM 配置字典，可选字段：
            - api_key: API Key（默认从 LLM_API_KEY 环境变量读取）
            - api_base: API 地址（默认从 LLM_API_BASE 环境变量读取）
            - model: 模型名（默认 gpt-4o-mini）
            - max_samples: 最多评测条数（默认 20）
            - max_workers: 并发数（默认 5）

    Returns:
        包含所有指标结果的 dict
    """
    results: Dict[str, Any] = {}

    if enable_rouge:
        rouge = compute_rouge(preds, refs)
        results.update(rouge)
        if rouge:
            print(f"  ROUGE-1: {rouge.get('rouge1', 0):.4f} | "
                  f"ROUGE-2: {rouge.get('rouge2', 0):.4f} | "
                  f"ROUGE-L: {rouge.get('rougeL', 0):.4f}")

    if enable_bertscore:
        bertscore = compute_bertscore(preds, refs)
        results.update(bertscore)
        if bertscore and "bertscore_f1" in bertscore:
            print(f"  BERTScore F1: {bertscore['bertscore_f1']:.4f}")

    if enable_llm_judge:
        llm_config = llm_config or {}
        llm_scores = compute_llm_judge(
            preds=preds,
            refs=refs,
            articles=articles,
            api_key=llm_config.get("api_key"),
            api_base=llm_config.get("api_base"),
            model=llm_config.get("model", "gpt-4o-mini"),
            max_samples=llm_config.get("max_samples", 20),
            max_workers=llm_config.get("max_workers", 5),
        )
        results.update(llm_scores)
        if llm_scores and "llm_avg" in llm_scores:
            print(f"  LLM Judge 平均分: {llm_scores['llm_avg']:.2f}/5")

    return results
