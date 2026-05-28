# ============================================================================
# 手动安装依赖（如果自动安装失败，可以运行这个脚本）
# ============================================================================

Write-Host "========================================" -ForegroundColor Green
Write-Host "T5 新闻摘要生成 - 依赖安装" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# 设置清华源镜像
$MIRROR = "-i https://pypi.tuna.tsinghua.edu.cn/simple"

Write-Host "正在安装必要依赖包..." -ForegroundColor Yellow
Write-Host ""

# 安装核心依赖
Write-Host "[1/6] 安装 datasets..." -ForegroundColor Cyan
pip install datasets $MIRROR

Write-Host ""
Write-Host "[2/6] 安装 transformers..." -ForegroundColor Cyan
pip install transformers $MIRROR

Write-Host ""
Write-Host "[3/6] 安装 torch (CPU版本)..." -ForegroundColor Cyan
pip install torch --index-url https://download.pytorch.org/whl/cpu

Write-Host ""
Write-Host "[4/6] 安装 rouge-score..." -ForegroundColor Cyan
pip install rouge-score $MIRROR

Write-Host ""
Write-Host "[5/6] 安装 matplotlib..." -ForegroundColor Cyan
pip install matplotlib $MIRROR

Write-Host ""
Write-Host "[6/6] 安装 tqdm..." -ForegroundColor Cyan
pip install tqdm $MIRROR

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "所有依赖安装完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "现在可以运行实验脚本了:" -ForegroundColor Yellow
Write-Host "  .\run_baseline_quick.ps1" -ForegroundColor White
Write-Host ""
