# 敏捷调试脚本：仅跑 20 条数据验证训练管线是否通畅
python train.py \
    --exp_id "Debug-QuickRun" \
    --lr 3e-4 \
    --batch_size 2 \
    --max_train_samples 20 \
    --epochs 1