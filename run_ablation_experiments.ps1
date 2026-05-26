# ============================================================================
# T5 新闻摘要生成 - 消融实验批量执行脚本
# ============================================================================
# 【使用说明】
# 1. 确保已安装所有依赖：pip install transformers datasets torch rouge-score matplotlib
# 2. 在 PowerShell 中运行：.\run_ablation_experiments.ps1
# 3. 所有实验结果保存在 t5-news-checkpoint/ 目录下
# 4. 每个实验会生成独立的时间戳可视化文件夹
#
# 【配置说明】
# - 训练集最大样本数：80
# - 验证集最大样本数：20
# - 训练轮数：5 epochs（可根据需要调整）
# - 模型：google-t5/t5-small
# ============================================================================

# 设置 Hugging Face 镜像加速
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 颜色输出函数
function Write-ColorOutput {
    param($Message, $Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

Write-ColorOutput "=" -Color Green
Write-ColorOutput "T5 新闻摘要生成 - 消融实验批量执行" -Color Green
Write-ColorOutput "=" -Color Green
Write-ColorOutput ""

# 步骤 1: 检查并安装依赖
Write-ColorOutput "步骤 1: 检查环境依赖..." -Color Yellow
Write-ColorOutput ""

$required_modules = @("datasets", "transformers", "torch", "rouge-score", "matplotlib", "tqdm")
$missing_modules = @()

foreach ($module in $required_modules) {
    try {
        python -c "import $module" 2>$null
        if ($LASTEXITCODE -ne 0) {
            $missing_modules += $module
        }
    } catch {
        $missing_modules += $module
    }
}

if ($missing_modules.Count -gt 0) {
    Write-ColorOutput "发现缺失的依赖包:" -Color Yellow
    $missing_modules | ForEach-Object { Write-ColorOutput "  - $_" -Color White }
    Write-ColorOutput ""
    Write-ColorOutput "正在安装... (使用清华源加速)" -Color Cyan
    
    $install_cmd = "pip install " + ($missing_modules -join " ") + " -i https://pypi.tuna.tsinghua.edu.cn/simple"
    Invoke-Expression $install_cmd
    
    Write-ColorOutput ""
    Write-ColorOutput "依赖安装完成！" -Color Green
} else {
    Write-ColorOutput "所有依赖已安装，跳过检查" -Color Green
}

Write-ColorOutput ""
Write-ColorOutput "步骤 2: 开始实验..." -Color Yellow
Write-ColorOutput ""

# 基础配置
$BASE_DIR = "t5-news-checkpoint"
$MAX_TRAIN = 80
$MAX_VAL = 20
$EPOCHS = 5

Write-ColorOutput "配置: 训练集=$MAX_TRAIN, 验证集=$MAX_VAL, Epochs=$EPOCHS" -Color Yellow
Write-ColorOutput ""

# 创建主输出目录
if (!(Test-Path $BASE_DIR)) {
    New-Item -ItemType Directory -Path $BASE_DIR | Out-Null
}

# ============================================================================
# 实验 1: 基准实验 (Baseline)
# ============================================================================
Write-ColorOutput "[1/10] 基准实验 (Baseline)..." -Color Cyan
Write-ColorOutput "      配置: lr=0.0003, batch_size=4, grad_accum=2" -Color Gray

python train.py `
    --max_train_samples $MAX_TRAIN `
    --max_val_samples $MAX_VAL `
    --epochs $EPOCHS `
    --lr 0.0003 `
    --batch_size 4 `
    --grad_accum 2 `
    --output_dir "$BASE_DIR/baseline"

Write-ColorOutput ""

# ============================================================================
# 实验 2: 学习率影响 - 低学习率
# ============================================================================
Write-ColorOutput "[2/10] 学习率实验 - 低学习率 (lr=0.0001)..." -Color Cyan

python train.py `
    --max_train_samples $MAX_TRAIN `
    --max_val_samples $MAX_VAL `
    --epochs $EPOCHS `
    --lr 0.0001 `
    --batch_size 4 `
    --grad_accum 2 `
    --output_dir "$BASE_DIR/exp_lr_1e4"

Write-ColorOutput ""

# ============================================================================
# 实验 3: 学习率影响 - 高学习率
# ============================================================================
Write-ColorOutput "[3/10] 学习率实验 - 高学习率 (lr=0.001)..." -Color Cyan

python train.py `
    --max_train_samples $MAX_TRAIN `
    --max_val_samples $MAX_VAL `
    --epochs $EPOCHS `
    --lr 0.001 `
    --batch_size 4 `
    --grad_accum 2 `
    --output_dir "$BASE_DIR/exp_lr_1e3"

Write-ColorOutput ""

# ============================================================================
# 实验 4: Batch Size 影响 - 小 Batch
# ============================================================================
Write-ColorOutput "[4/10] Batch Size 实验 - 小Batch (bs=2, ga=4)..." -Color Cyan

python train.py `
    --max_train_samples $MAX_TRAIN `
    --max_val_samples $MAX_VAL `
    --epochs $EPOCHS `
    --lr 0.0003 `
    --batch_size 2 `
    --grad_accum 4 `
    --output_dir "$BASE_DIR/exp_bs2_ga4"

Write-ColorOutput ""

# ============================================================================
# 实验 5: Batch Size 影响 - 大 Batch
# ============================================================================
Write-ColorOutput "[5/10] Batch Size 实验 - 大Batch (bs=8, ga=1)..." -Color Cyan

python train.py `
    --max_train_samples $MAX_TRAIN `
    --max_val_samples $MAX_VAL `
    --epochs $EPOCHS `
    --lr 0.0003 `
    --batch_size 8 `
    --grad_accum 1 `
    --output_dir "$BASE_DIR/exp_bs8_ga1"

Write-ColorOutput ""

# ============================================================================
# 实验 6: 序列长度 - 短序列
# ============================================================================
Write-ColorOutput "[6/10] 序列长度实验 - 短序列 (256/30)..." -Color Cyan

python train.py `
    --max_train_samples $MAX_TRAIN `
    --max_val_samples $MAX_VAL `
    --epochs $EPOCHS `
    --lr 0.0003 `
    --batch_size 4 `
    --grad_accum 2 `
    --max_source_len 256 `
    --max_target_len 30 `
    --output_dir "$BASE_DIR/exp_len_short"

Write-ColorOutput ""

# ============================================================================
# 实验 7: 序列长度 - 长序列
# ============================================================================
Write-ColorOutput "[7/10] 序列长度实验 - 长序列 (768/50)..." -Color Cyan

python train.py `
    --max_train_samples $MAX_TRAIN `
    --max_val_samples $MAX_VAL `
    --epochs $EPOCHS `
    --lr 0.0003 `
    --batch_size 4 `
    --grad_accum 2 `
    --max_source_len 768 `
    --max_target_len 50 `
    --output_dir "$BASE_DIR/exp_len_long"

Write-ColorOutput ""

# ============================================================================
# 实验 8: 训练策略 - 无梯度累积
# ============================================================================
Write-ColorOutput "[8/10] 训练策略实验 - 无梯度累积 (ga=1)..." -Color Cyan

python train.py `
    --max_train_samples $MAX_TRAIN `
    --max_val_samples $MAX_VAL `
    --epochs $EPOCHS `
    --lr 0.0003 `
    --batch_size 8 `
    --grad_accum 1 `
    --output_dir "$BASE_DIR/exp_no_accum"

Write-ColorOutput ""

# ============================================================================
# 实验 9: 数据量影响 - 少量数据
# ============================================================================
Write-ColorOutput "[9/10] 数据量实验 - 少量数据 (40 samples)..." -Color Cyan

python train.py `
    --max_train_samples 40 `
    --max_val_samples 10 `
    --epochs $EPOCHS `
    --lr 0.0003 `
    --batch_size 4 `
    --grad_accum 2 `
    --output_dir "$BASE_DIR/exp_data_40"

Write-ColorOutput ""

# ============================================================================
# 实验 10: 数据量影响 - 中等数据
# ============================================================================
Write-ColorOutput "[10/10] 数据量实验 - 中等数据 (120 samples)..." -Color Cyan

python train.py `
    --max_train_samples 120 `
    --max_val_samples 30 `
    --epochs $EPOCHS `
    --lr 0.0003 `
    --batch_size 4 `
    --grad_accum 2 `
    --output_dir "$BASE_DIR/exp_data_120"

Write-ColorOutput ""

# ============================================================================
# 实验完成总结
# ============================================================================
Write-ColorOutput "=" -Color Green
Write-ColorOutput "所有实验完成！" -Color Green
Write-ColorOutput "=" -Color Green
Write-ColorOutput ""
Write-ColorOutput "实验结果保存在: $BASE_DIR/" -Color Yellow
Write-ColorOutput ""
Write-ColorOutput "实验列表:" -Color Cyan
Write-ColorOutput "  1. baseline/           - 基准配置 (lr=0.0003, bs=4, ga=2)" -Color White
Write-ColorOutput "  2. exp_lr_1e4/         - 低学习率 (lr=0.0001)" -Color White
Write-ColorOutput "  3. exp_lr_1e3/         - 高学习率 (lr=0.001)" -Color White
Write-ColorOutput "  4. exp_bs2_ga4/        - 小Batch (bs=2, ga=4)" -Color White
Write-ColorOutput "  5. exp_bs8_ga1/        - 大Batch (bs=8, ga=1)" -Color White
Write-ColorOutput "  6. exp_len_short/      - 短序列 (256/30)" -Color White
Write-ColorOutput "  7. exp_len_long/       - 长序列 (768/50)" -Color White
Write-ColorOutput "  8. exp_no_accum/       - 无梯度累积 (ga=1)" -Color White
Write-ColorOutput "  9. exp_data_40/        - 少量数据 (40 samples)" -Color White
Write-ColorOutput " 10. exp_data_120/       - 中等数据 (120 samples)" -Color White
Write-ColorOutput ""
Write-ColorOutput "提示:" -Color Yellow
Write-ColorOutput "  - 每个实验目录下都有 visualization_时间戳/ 文件夹包含可视化图表" -Color Gray
Write-ColorOutput "  - 查看 train_hparams.json 了解每个实验的具体配置" -Color Gray
Write-ColorOutput "  - 对比各实验的 ROUGE 分数和 Loss 曲线进行分析" -Color Gray
Write-ColorOutput ""
