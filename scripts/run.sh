MODELS=('PowerGCT')

## 训练
for param in "${MODELS[@]}"; do
    python -u train.py \
              --task_name classification \
              --model "$param" \
              --checkpoints ./checkpoints \
              --train_root_path ./datasets/train \
              --test_root_path ./datasets/test \
              --learning_rate 0.001 \
              --batch_size 16 \
              --train_epochs 30
done

## 推理
for param in "${MODELS[@]}"; do
    python -u inference.py \
              --model "$param" \
              --task_name classification \
              --test_root_path ./datasets/test \
              --checkpoints ./checkpoints \
              --result_save_path ./results
done
