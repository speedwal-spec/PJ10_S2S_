"""
Gradio Web界面模块
职责：构建A/B测试竞技场UI，提供交互式模型对比功能
"""
import gradio as gr
import os
import json
from typing import Tuple
from core.model_manager import ModelManager
from configs.config_manager import load_full_config


# ==========================================
# 1. 初始化全局模型管理器（单例，由 launch() 延迟初始化）
# ==========================================
manager = None


# ==========================================
# 2. UI 前后端桥接函数 (Bridge Callbacks)
# ==========================================

def _fmt(val, fmt_float="{}", fmt_int="{}"):
    """智能格式化：浮点数/整数/字符串统一处理，None 返回 N/A"""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return fmt_float.format(val)
    if isinstance(val, int):
        return fmt_int.format(val)
    return str(val)


def _generate_card_md(selected_exp_id: str) -> str:
    """
    生成模型属性卡片Markdown
    
    Args:
        selected_exp_id: 选中的实验ID
        
    Returns:
        Markdown格式的模型信息
    """
    if not selected_exp_id: 
        return ""
    
    profile = manager.registry.get(selected_exp_id, {})
    hparams = profile.get("hparams", {})
    metrics = profile.get("metrics", {})
    desc = profile.get("description", "")
    path = profile.get("path", "")
    
    # ── 摘要描述 ──
    card_md = f"**📋 模型概要**\n"
    if desc:
        card_md += f"> {desc}\n\n"
    card_md += f"- 实验ID: `{selected_exp_id}`\n"
    card_md += f"- 模型路径: `{path}`\n\n"
    
    # ── 超参数 ──
    card_md += "**🔬 训练超参**\n"
    
    lr_val = hparams.get('lr')
    lr_str = _fmt(lr_val, fmt_float="{:g}")  # 科学计数法自动选择
    bs_str = _fmt(hparams.get('batch_size'))
    ga_str = _fmt(hparams.get('grad_accum'))
    ep_str = _fmt(hparams.get('epochs'))
    warm_str = _fmt(hparams.get('warmup_ratio'), fmt_float="{:.0%}")
    seed_str = _fmt(hparams.get('seed'))
    
    card_md += f"- 学习率: `{lr_str}` | 批大小: `{bs_str}` | 梯度累积: `{ga_str}`\n"
    card_md += f"- 训练轮数: `{ep_str}` | warmup: `{warm_str}` | seed: `{seed_str}`\n"
    
    data_str = _fmt(hparams.get('max_train_samples'))
    val_str = _fmt(hparams.get('max_val_samples'))
    src_str = _fmt(hparams.get('max_source_len'))
    tgt_str = _fmt(hparams.get('max_target_len'))
    card_md += f"- 训练数据: `{data_str}` 条 | 验证数据: `{val_str}` 条\n"
    card_md += f"- 源文截断: `{src_str}` | 生成截断: `{tgt_str}`\n\n"
    
    # ── ROUGE 评测 ──
    card_md += "**🏆 客观评测 (ROUGE)**\n"
    if metrics:
        r1 = metrics.get('rouge1', 0)
        r2 = metrics.get('rouge2', 0)
        rl = metrics.get('rougeL', 0)
        card_md += f"- ROUGE-1: `{r1:.4f}` | ROUGE-2: `{r2:.4f}` | ROUGE-L: `{rl:.4f}`\n"
    else:
        card_md += "- *暂无离线评测数据（可运行 evaluate_rouge.py 补充）*\n"
    
    return card_md


def on_model_change_a(selected_exp_id: str) -> Tuple[str, str]:
    """
    卡槽A模型切换回调
    
    Args:
        selected_exp_id: 选中的实验ID
        
    Returns:
        (状态消息, 模型卡片Markdown)
    """
    if not selected_exp_id: 
        return "请选择模型", ""
    
    status = manager.load_model(selected_exp_id, slot="a")
    return status, _generate_card_md(selected_exp_id)


def on_model_change_b(selected_exp_id: str) -> Tuple[str, str]:
    """
    卡槽B模型切换回调
    
    Args:
        selected_exp_id: 选中的实验ID
        
    Returns:
        (状态消息, 模型卡片Markdown)
    """
    if not selected_exp_id: 
        return "请选择模型", ""
    
    status = manager.load_model(selected_exp_id, slot="b")
    return status, _generate_card_md(selected_exp_id)


def on_submit_dual(article: str, max_len: int, penalty: float) -> Tuple[str, str]:
    """
    双模型同步生成回调
    
    Args:
        article: 输入文章
        max_len: 最大生成长度
        penalty: 长度惩罚系数
        
    Returns:
        (卡槽A输出, 卡槽B输出)
    """
    out_a = manager.generate_slot(article, max_len, penalty, slot="a")
    out_b = manager.generate_slot(article, max_len, penalty, slot="b")
    return out_a, out_b



# ==========================================
# 3. 渲染 (Theme) — 深紫蓝暗色主题
# ==========================================
# 配色策略：
#   body_background  = #1e1e3a  暗紫蓝背景（中深，给容器留变暗空间）
#   block_background = #161630  容器比背景更深
#   code background  = #2d2d52  代码块背景（CSS 注入）
#   code text        = #f0d060  暖金色（深色底上最清晰）

custom_css = """
/* ── 修复 Markdown 行内代码（backtick `` 渲染的 <code>）── */
.prose :not(pre) > code,
.markdown-body code {
    background-color: #2d2d52 !important;
    color: #f0d060 !important;
    font-size: 0.88em !important;
    padding: 0.15em 0.4em !important;
    border-radius: 4px !important;
    border: 1px solid #4a4a70 !important;
}

/* ── 修复输出框文字 —— */
.gr-box textarea, .gr-box input, .gr-textarea textarea {
    color: #e0e0e0 !important;
}

/* ── 修复快捷测试样例文字（深色底 + 黑字问题）──
   覆盖 Gradio v3/v4/v5 不同版本的 class 命名差异
   Gradio 4+ 每行是 <button> 而不是纯 <td> 文字 */
.gr-sample-table td,
.gr-sample-table td *,
.gr-sample-table td span,
table.examples td,
table.examples td span,
.examples-table td,
.examples-table td span,
.examples td,
.examples td *,
td.examples-cell,
span.examples-label,
.examples .label,
.gr-sample-table button,
.gr-sample-table button span,
table.examples button,
table.examples button span,
.examples-table button,
.examples-table button span {
    color: #d0d0e0 !important;
}
.gr-sample-table tr:hover td,
.examples tr:hover td,
.examples-table tr:hover td,
tr:hover td.examples-cell {
    color: #ffffff !important;
}

/* ── 快捷测试示例区域加浅色底框，与深色容器区分 ── */
.gr-sample-table,
table.examples,
.examples-table > table,
.examples-table,
.examples > table,
.examples-container,
div.examples {
    background-color: #2d2d52 !important;
    border-radius: 8px !important;
}

/* ── Dropdown / Slider 标签文字颜色 ── */
.gr-label, label, .gr-form-label {
    color: #c8c8d8 !important;
}

/* ── 状态文本框中文字 ── */
textarea[readonly], input[readonly] {
    color: #b0b0c0 !important;
}
"""

gemini_theme = gr.themes.Default(
    font=[gr.themes.GoogleFont("Google Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    primary_hue="blue",
    neutral_hue="slate",
    radius_size="lg",
).set(
    # ── 全局底色（暗紫蓝）与字体色 ──
    body_background_fill="#1e1e3a",
    body_text_color="#e8e8e8",
    
    # ── 浮动层与下拉菜单底色（比背景更深）──
    background_fill_primary="#161630",
    background_fill_secondary="#121228",

    # ── 容器与边框 ──
    block_background_fill="#161630",
    block_border_width="1px",
    block_border_color="#4a4a70",
    block_label_background_fill="#161630",
    block_label_text_color="#c8c8d8",
    
    # ── 输入框本体 ──
    input_background_fill="#252548",
    input_border_color="#5a5a80",
    
    # ── 按钮与滑动条 ──
    button_primary_background_fill="#c4c7c5",
    button_primary_background_fill_hover="#ffffff",
    button_primary_text_color="#131314",
    button_primary_text_color_hover="#131314",
    slider_color="#7c6ff0",
)


# ==========================================
# 4. 搭建响应式 Web 界面 (A/B Testing Arena)
# ==========================================

def create_demo(cfg) -> gr.Blocks:
    """
    创建Gradio演示应用
    
    Args:
        cfg: 实验配置，用于推理默认值和路径
        
    Returns:
        Gradio Blocks应用
    """
    with gr.Blocks(title="Neural News MLOps", theme=gemini_theme, css=custom_css) as demo:
        
        gr.Markdown(
            """
            <div style="text-align: center; padding: 25px 0 10px 0;">
                <h1 style="font-weight: 400; letter-spacing: 0.5px; font-size: 34px; margin-bottom: 5px;">📰 Neural News Summarizer</h1>
                <p style="color: #8ab4f8; font-size: 15px; font-weight: 500;">Powered by Auto-MLOps & A/B Evaluation Arena</p>
            </div>
            """
        )
        
        model_choices = manager.list_models()
        default_model = model_choices[0] if model_choices else None
        default_b = model_choices[1] if len(model_choices) > 1 else default_model

        with gr.Row():
            # --- 左侧：全局数据源与公共参数 ---
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 📥 共享数据流入口")
                    input_text = gr.Textbox(
                        lines=10, 
                        label="原始正文 (Shared Raw Text Source)", 
                        placeholder="在此处粘贴冗长的英文新闻报道..."
                    )
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ 清空画布", variant="secondary")
                        submit_btn = gr.Button("✨ 双模同步提炼标题", variant="primary")
                
                with gr.Accordion("⚙️ 全局推理引擎参数", open=True):
                    max_len_slider = gr.Slider(minimum=10, maximum=100, value=cfg.inference.max_new_tokens, step=1, 
                                              label="生成截断长度 (Max Tokens)")
                    penalty_slider = gr.Slider(minimum=0.1, maximum=2.0, value=cfg.inference.length_penalty, step=0.05, 
                                              label="长度惩罚 (Length Penalty, <1 偏短句)")

            # --- 右侧：消融模型 ---
            with gr.Column(scale=2):
                gr.Markdown("### ⚔️ 消融实验 A/B 对比舱 (Evaluation Arena)")
                
                with gr.Row():
                    # 🔴  A：对照组
                    with gr.Column(scale=1, variant="panel"):
                        gr.Markdown("#### 🔴 模型 A")
                        model_a_dropdown = gr.Dropdown(choices=model_choices, value=default_model, 
                                                      label="选择挂载的模型 A")
                        status_a = gr.Textbox(label="引擎 A 状态", value="等待挂载...", 
                                            interactive=False, lines=1)
                        model_card_a = gr.Markdown("*(加载后显示属性)*")
                        output_text_a = gr.Textbox(
                            lines=6, 
                            label="📡 模型 A 提炼产出",
                            placeholder="模型 A 的生成标题将显示在这里..."
                        )

                    # 🔵  B：实验组
                    with gr.Column(scale=1, variant="panel"):
                        gr.Markdown("#### 🔵 模型 B")
                        model_b_dropdown = gr.Dropdown(choices=model_choices, value=default_b, 
                                                      label="选择挂载的模型 B")
                        status_b = gr.Textbox(label="引擎 B 状态", value="等待挂载...", 
                                            interactive=False, lines=1)
                        model_card_b = gr.Markdown("*(加载后显示属性)*")
                        output_text_b = gr.Textbox(
                            lines=6, 
                            label="📡 模型 B 提炼产出",
                            placeholder="模型 B 的生成标题将显示在这里..."
                        )

        # ==========================================
        # 5. 动态加载真实测试集样例
        # ==========================================
        example_list = []
        sample_file = cfg.paths.samples_file
        if os.path.exists(sample_file):
            with open(sample_file, "r", encoding="utf-8") as f:
                samples = json.load(f).get("items", [])
                for item in samples[:5]:
                    example_list.append([item["article"]])
        
        if not example_list:
            example_list = [
                ["(CNN) -- A magnitude 7.8 earthquake has struck Nepal, causing immense destruction in the capital, Kathmandu, and triggering deadly avalanches on Mount Everest. Rescuers are frantically digging through the rubble of collapsed buildings to find survivors. The death toll is expected to rise significantly as more remote areas are reached."],
                ["Apple Inc. announced on Tuesday that it will transition its entire Mac lineup from Intel processors to its own custom-designed ARM-based silicon, a move that promises significant performance and energy efficiency improvements."]
            ]

        gr.Examples(
            examples=example_list,
            inputs=input_text,
            label="⚡ 快捷测试数据 (动态抓取自 Test 集抽样)",
        )

        # ==========================================
        # 6. 事件绑定与回调监听
        # ==========================================
        model_a_dropdown.change(fn=on_model_change_a, inputs=[model_a_dropdown], 
                               outputs=[status_a, model_card_a])
        model_b_dropdown.change(fn=on_model_change_b, inputs=[model_b_dropdown], 
                               outputs=[status_b, model_card_b])
        
        # 初始化启动时，自动触发展示与加载机制
        if default_model:
            demo.load(fn=on_model_change_a, inputs=[gr.State(default_model)], 
                     outputs=[status_a, model_card_a])
        if default_b:
            demo.load(fn=on_model_change_b, inputs=[gr.State(default_b)], 
                     outputs=[status_b, model_card_b])

        submit_btn.click(
            fn=on_submit_dual, 
            inputs=[input_text, max_len_slider, penalty_slider], 
            outputs=[output_text_a, output_text_b]
        )
        
        clear_btn.click(
            fn=lambda: ("", "", ""), 
            inputs=None, 
            outputs=[input_text, output_text_a, output_text_b]
        )

    return demo


# ==========================================
# 7. 启动入口
# ==========================================
def launch(server_name: str = "127.0.0.1", server_port: int = 7860, 
           share: bool = False):
    """
    启动Gradio Web服务
    
    Args:
        server_name: 服务器地址
        server_port: 服务器端口
        share: 是否创建公开分享链接
    """
    global manager
    print("🚀 MLOps Web Gateway 启动中...")
    cfg = load_full_config()
    manager = ModelManager(
        ablation_dir=cfg.paths.ablation_base,
        legacy_dir=cfg.paths.output_dir,
        config=cfg
    )
    demo = create_demo(cfg)
    demo.launch(server_name=server_name, server_port=server_port, share=share)


if __name__ == "__main__":
    launch()
