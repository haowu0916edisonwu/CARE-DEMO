#!/bin/bash
# ---------------------------------------------------------
# CARE Llama-3-8B 最终稳定版 (修复 404 路径 + 自动化 CP)
# ---------------------------------------------------------

# 1. 路径配置
PROJECT_ROOT="/mnt/care_workspace/CARE"
LOG_FILE="/mnt/care_workspace/logs/llama_run.log"
OUTPUT_DIR="/mnt/care_workspace/outputs"
CKPT_FINAL="/mnt/care_workspace/checkpoints/llama3"
# 使用图像中确认的 NousResearch 全量版本
MODEL_ID="NousResearch/Meta-Llama-3-8B-Instruct"
# 修正路径：根据 image_02964e.jpg，正确仓库名为 care_llama_pt
AUTH_CKPT="eunseong/care_llama_pt"

# 2. 清理与初始化
pkill -9 python
pkill -9 accelerate
mkdir -p /mnt/care_workspace/logs "$OUTPUT_DIR/checkpoint" "$CKPT_FINAL"
rm -rf "$OUTPUT_DIR"/checkpoint/*

cd "$PROJECT_ROOT"

# 3. 物理修复 (路径与参数)
sed -i 's|checkpoint_dir = \[os.path.join(args.workdir, "checkpoint")\]|checkpoint_dir = ["/mnt/care_workspace/outputs/checkpoint"]|g' src/language_modeling/train.py
sed -i 's/lora_r: .*/lora_r: 64/g' config/language_modeling/finetune.yaml
sed -i 's/lora_alpha: .*/lora_alpha: 128/g' config/language_modeling/finetune.yaml

# 4. 后台挂载启动
echo "[$(date +%T)] 正在启动 Llama-3 任务，请确保已执行 huggingface-cli login..."
export WANDB_MODE=disabled

(
  # A. 执行训练
  accelerate launch \
    --mixed_precision bf16 --num_machines 1 --num_processes 1 \
    -m src.language_modeling.train \
    --config config/language_modeling/finetune.yaml \
    --model_name_or_path "$MODEL_ID" \
    --train_file "data_care/finetune/nq_llama.jsonl" \
    --dev_file "data_care/finetune/nq_valid.jsonl" \
    --checkpoint_path "$AUTH_CKPT" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-4 \
    --num_train_epochs 2 \
    --exp_name "repro_llama3_final" \
    --workdir "$OUTPUT_DIR" \
    --checkpointing_steps 300

  # 5. 权重搬运与评测
  if [ -d "$OUTPUT_DIR/checkpoint" ] && [ "$(ls -A $OUTPUT_DIR/checkpoint)" ]; then
      cp -r "$OUTPUT_DIR/checkpoint"/* "$CKPT_FINAL/"
      python3 -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('$MODEL_ID'); t.save_pretrained('$CKPT_FINAL')"
      python -m src.eval.run_eval --data nq --checkpoint_path "$CKPT_FINAL" --use_rag --eval_batch_size 8 --save_results --results_path "eval_result_llama3_nq.json"
  fi
) >> "$LOG_FILE" 2>&1 &

echo "===================================================="
echo "自动化任务已启动！监控：tail -f $LOG_FILE"
echo "===================================================="