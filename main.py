import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import json
import gc

# ==========================================
# 1. 核心架构：多槽位模型管理器 (Dual-Slot Model Manager)
# 作用：负责动态扫描硬盘资产、管理双槽位模型热切换、防止显存溢出
# ==========================================
class ModelManager:
    def __init__(self, ablation_dir="checkpoints_ablation", legacy_dir="t5-news-checkpoint"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        )
        
        # 分离 A/B 两个卡槽的模型与状态
        self.model_a, self.tokenizer_a, self.current_model_a = None, None, None
        self.model_b, self.tokenizer_b, self.current_model_b = None, None, None
        
        # 扫描并注册硬盘上的可用模型资产
        self.registry = self._scan_assets(ablation_dir, legacy_dir)
        print(f"📦 [Model Hub] 共扫描到 {len(self.registry)} 个可用模型资产。")

    def _scan_assets(self, ablation_dir, legacy_dir):
        registry = {}
        if os.path.exists(ablation_dir):
            for exp_id in os.listdir(ablation_dir):
                model_path = os.path.join(ablation_dir, exp_id)
                if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
                    registry[exp_id] = self._build_model_profile(exp_id, model_path)
        
        if os.path.exists(legacy_dir) and os.path.exists(os.path.join(legacy_dir, "config.json")):
            if "Legacy-Baseline" not in registry:
                registry["Legacy-Baseline"] = self._build_model_profile("Legacy-Baseline", legacy_dir)
                
        return registry

    def _build_model_profile(self, exp_id, model_path):
        profile = {"path": model_path, "hparams": {}, "metrics": {}}
        hparams_path = os.path.join(model_path, "train_hparams.json")
        if os.path.exists(hparams_path):
            with open(hparams_path, "r", encoding="utf-8") as f:
                profile["hparams"] = json.load(f)
                
        metrics_path = f"results_{exp_id}.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                profile["metrics"] = json.load(f)
        return profile

    def load_model(self, exp_id, slot="a"):
        """显存安全的双槽热切换：先释放旧模型，再加载新模型，避免双倍峰值 OOM"""
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
        
        # 🔴 终极绝杀：关闭低内存模式，手动缝合权重，再安全推入显卡
        net = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=False  # 明确拒绝 accelerate 的虚拟张量加载
        )
        
        # 🌟 魔法指令：强行把 T5 分离的 encoder、decoder 和 lm_head 权重绑定到一起
        net.tie_weights() 
        
        # 缝合完毕后，作为一个完整的实体，堂堂正正地推入显卡
        net = net.to(self.device)
        net.eval()

        # 3. 绑定到指定卡槽
        if slot == "a":
            self.model_a, self.tokenizer_a, self.current_model_a = net, tok, exp_id
        else:
            self.model_b, self.tokenizer_b, self.current_model_b = net, tok, exp_id
            
        print(f"✅ 卡槽 {slot.upper()} 挂载成功！")
        return f"✅ 成功挂载至卡槽 {slot.upper()}"

    def generate_slot(self, article, max_length, length_penalty, slot="a"):
        """指定卡槽进行独立推理"""
        model = self.model_a if slot == "a" else self.model_b
        tokenizer = self.tokenizer_a if slot == "a" else self.tokenizer_b
        curr_id = self.current_model_a if slot == "a" else self.current_model_b
        
        if not model: return f"⚠️ 卡槽 {slot.upper()} 尚未挂载模型！"
        if not article.strip(): return ""
        
        prefix = self.registry[curr_id]["hparams"].get("prefix", "summarize: ")
        inputs = tokenizer(
            prefix + article, 
            max_length=512, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_length, 
                num_beams=4,
                length_penalty=length_penalty, 
                early_stopping=True
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 全局单例实例化
manager = ModelManager()

# ==========================================
# 2. UI 前后端桥接函数 (Bridge Callbacks)
# ==========================================
def _generate_card_md(selected_exp_id):
    if not selected_exp_id: return ""
    profile = manager.registry.get(selected_exp_id, {})
    hparams = profile.get("hparams", {})
    metrics = profile.get("metrics", {})
    
    card_md = f"**🔬 训练超参**\n"
    card_md += f"- 学习率: `{hparams.get('lr', 'N/A')}` | 批大小: `{hparams.get('batch_size', 'N/A')}`\n"
    card_md += f"- 训练数据: `{hparams.get('max_train_samples', 'N/A')}` 条\n\n"
    
    card_md += f"**🏆 客观评测 (ROUGE)**\n"
    if metrics:
        card_md += f"- ROUGE-1: `{metrics.get('rouge1', 0):.4f}`\n"
        card_md += f"- ROUGE-L: `{metrics.get('rougeL', 0):.4f}`"
    else:
        card_md += "- *暂无离线评测数据*"
    return card_md

def on_model_change_a(selected_exp_id):
    if not selected_exp_id: return "请选择模型", ""
    status = manager.load_model(selected_exp_id, slot="a")
    return status, _generate_card_md(selected_exp_id)

def on_model_change_b(selected_exp_id):
    if not selected_exp_id: return "请选择模型", ""
    status = manager.load_model(selected_exp_id, slot="b")
    return status, _generate_card_md(selected_exp_id)

def on_submit_dual(article, max_len, penalty):
    out_a = manager.generate_slot(article, max_len, penalty, slot="a")
    out_b = manager.generate_slot(article, max_len, penalty, slot="b")
    return out_a, out_b


# 3. 极客级暗黑主题渲染 (Theme)
# ==========================================
gemini_theme = gr.themes.Default(
    font=[gr.themes.GoogleFont("Google Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    primary_hue="blue",
    neutral_hue="slate",
    radius_size="lg",
).set(
    # --- 全局底色与字体色 ---
    body_background_fill="#131314", 
    body_text_color="#e3e3e3",          # 这个颜色会自动安全地继承到所有输入框和下拉菜单的文字上
    
    # --- 核心修复：定义浮动层与下拉菜单底色 ---
    background_fill_primary="#1e1f20",  # 彻底解决下拉菜单白底刺眼的问题
    background_fill_secondary="#131314",

    # --- 容器与边框 ---
    block_background_fill="#1e1f20", 
    block_border_width="1px", 
    block_border_color="#444746",
    block_label_background_fill="#1e1f20", 
    block_label_text_color="#e3e3e3",
    
    # --- 输入框本体 ---
    input_background_fill="#1e1f20", 
    input_border_color="#444746",
    
    # --- 按钮与滑动条 ---
    button_primary_background_fill="#c4c7c5", 
    button_primary_background_fill_hover="#ffffff",
    button_primary_text_color="#131314", 
    button_primary_text_color_hover="#131314",
    slider_color="#8ab4f8",
)
# ==========================================
# 4. 搭建响应式 Web 界面 (A/B Testing Arena)
# ==========================================
with gr.Blocks(title="Neural News MLOps") as demo:
    
    gr.Markdown(
        """
        <div style="text-align: center; padding: 25px 0 10px 0;">
            <h1 style="font-weight: 400; letter-spacing: 0.5px; font-size: 34px; margin-bottom: 5px;">📰 Neural News Summarizer</h1>
            <p style="color: #8ab4f8; font-size: 15px; font-weight: 500;">Powered by Auto-MLOps & A/B Evaluation Arena</p>
        </div>
        """
    )
    
    model_choices = list(manager.registry.keys())
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
                max_len_slider = gr.Slider(minimum=10, maximum=100, value=40, step=1, label="生成截断长度 (Max Tokens)")
                penalty_slider = gr.Slider(minimum=0.1, maximum=2.0, value=0.85, step=0.05, label="长度惩罚 (Length Penalty, <1 偏短句)")

        # --- 右侧：消融模型竞技场 ---
        with gr.Column(scale=2):
            gr.Markdown("### ⚔️ 消融实验 A/B 对比舱 (Evaluation Arena)")
            
            with gr.Row():
                # 🔴 舱位 A：对照组
                with gr.Column(scale=1, variant="panel"):
                    gr.Markdown("#### 🔴 候选模型 A")
                    model_a_dropdown = gr.Dropdown(choices=model_choices, value=default_model, label="选择挂载的模型 A")
                    status_a = gr.Textbox(label="引擎 A 状态", value="等待挂载...", interactive=False, lines=1)
                    model_card_a = gr.Markdown("*(加载后显示属性)*")
                    output_text_a = gr.Textbox(
                        lines=6, 
                        label="📡 模型 A 提炼产出",
                        placeholder="模型 A 的生成标题将显示在这里..."
                    )

                # 🔵 舱位 B：实验组
                with gr.Column(scale=1, variant="panel"):
                    gr.Markdown("#### 🔵 候选模型 B")
                    model_b_dropdown = gr.Dropdown(choices=model_choices, value=default_b, label="选择挂载的模型 B")
                    status_b = gr.Textbox(label="引擎 B 状态", value="等待挂载...", interactive=False, lines=1)
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
    sample_file = "sample_articles_20.json"
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
        label="⚡ 快捷测试数据 (动态抓取自 Test 集抽样)"
    )

    # ==========================================
    # 6. 事件绑定与回调监听
    # ==========================================
    model_a_dropdown.change(fn=on_model_change_a, inputs=[model_a_dropdown], outputs=[status_a, model_card_a])
    model_b_dropdown.change(fn=on_model_change_b, inputs=[model_b_dropdown], outputs=[status_b, model_card_b])
    
    # 初始化启动时，自动触发展示与加载机制
    if default_model:
        demo.load(fn=on_model_change_a, inputs=[gr.State(default_model)], outputs=[status_a, model_card_a])
    if default_b:
        demo.load(fn=on_model_change_b, inputs=[gr.State(default_b)], outputs=[status_b, model_card_b])

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

if __name__ == "__main__":
    print("🚀 MLOps Web Gateway 启动中...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False,theme=gemini_theme)