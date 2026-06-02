# ============================================================================
# T5 新闻摘要生成 - 快速开始（仅运行基准实验）
# ============================================================================
# 【适合首次运行】
# - 只运行基准实验，验证环境配置正确
# - 训练集: 80 条样本
# - 验证集: 20 条样本
# - 训练轮数: 5 epochs
# - 预计时间: 10-20 分钟（取决于你的电脑配置）
# ============================================================================

# 设置 Hugging Face 镜像加速
$env:HF_ENDPOINT = "https://hf-mirror.com"

Write-Host "========================================" -ForegroundColor Green
Write-Host "T5 新闻摘要生成 - 基准实验" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "步骤 1: 检查并安装依赖..." -ForegroundColor Yellow
Write-Host ""

# 检查并安装必要依赖
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
    Write-Host "发现缺失的依赖包:" -ForegroundColor Yellow
    $missing_modules | ForEach-Object { Write-Host "  - $_" -ForegroundColor White }
    Write-Host ""
    Write-Host "正在安装... (使用清华源加速)" -ForegroundColor Cyan
    
    $install_cmd = "pip install " + ($missing_modules -join " ") + " -i https://pypi.tuna.tsinghua.edu.cn/simple"
    Invoke-Expression $install_cmd
    
    Write-Host ""
    Write-Host "依赖安装完成！" -ForegroundColor Green
} else {
    Write-Host "所有依赖已安装，跳过检查" -ForegroundColor Green
}

Write-Host ""
Write-Host "步骤 2: 开始训练..." -ForegroundColor Yellow
Write-Host ""
Write-Host "配置信息:" -ForegroundColor Yellow
Write-Host "  训练集样本数: 80" -ForegroundColor White
Write-Host "  验证集样本数: 20" -ForegroundColor White
Write-Host "  训练轮数: 5 epochs" -ForegroundColor White
Write-Host "  学习率: 0.0003" -ForegroundColor White
Write-Host "  Batch Size: 4" -ForegroundColor White
Write-Host "  梯度累积: 2" -ForegroundColor White
Write-Host ""

# 运行基准实验
python train.py `
    --max_train_samples 80 `
    --max_val_samples 20 `
    --epochs 5 `
    --lr 0.0003 `
    --batch_size 4 `
    --grad_accum 2 `
    --output_dir "t5-news-checkpoint/baseline"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "基准实验完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "结果保存在: t5-news-checkpoint/baseline/" -ForegroundColor Yellow
Write-Host ""
Write-Host "下一步:" -ForegroundColor Cyan
Write-Host "  1. 查看生成的可视化图表" -ForegroundColor White
Write-Host "  2. 记录 ROUGE 分数作为基准" -ForegroundColor White
Write-Host "  3. 运行 run_ablation_experiments.ps1 进行完整实验" -ForegroundColor White
Write-Host ""
