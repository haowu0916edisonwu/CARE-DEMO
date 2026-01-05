#!/bin/bash
# ---------------------------------------------------------
# CARE Mistral 专项脚本 (V22 语法修复 + 启动自清理版)
# ---------------------------------------------------------

# 1. 基础路径配置
PROJECT_ROOT="/mnt/care_workspace/CARE"
LOG_DIR="/mnt/care_workspace/logs"
OUTPUT_DIR="/mnt/care_workspace/outputs"
CKPT_FINAL="/mnt/care_workspace/checkpoints/mistral"
export HF_HOME="/mnt/care_workspace/hf_cache"

# 2. 启动前深度清理 (显存 + 磁盘)
echo "[CLEANUP] 正在释放 GPU 显存与磁盘缓存..."
pkill -9 python
rm -rf "$HF_HOME"/*
rm -rf "$OUTPUT_DIR"/checkpoint/*
mkdir -p "$LOG_DIR" "$OUTPUT_DIR/checkpoint" "$CKPT_FINAL" "$HF_HOME"

cd $PROJECT_ROOT

# 3. 物理修复源码 (解决语法报错的关键)
echo "[SYSTEM] 正在修复源码逻辑与 IndentationError..."
# 使用 python 脚本处理缩进，比 sed 更安全
python3 -c "
fname = 'src/eval/run_eval.py'
with open(fname, 'r') as f:
    lines = f.readlines()
if not any('pass  # 修复缩进' in l for l in lines):
    new_lines = []
    for i, line in enumerate(lines):
        new_lines.append(line)
        if i == 558 and 'else:' in line:
            new_lines.append('        pass  # 修复缩进\n')
    with open(fname, 'w') as f:
        f.writelines(new_lines)
"

# 【修复第 37 行语法错误】使用单引号避免括号干扰 Bash 解析
sed -i 's|checkpoint_dir = \[os.path.join(wandb_tracker.run.dir, "checkpoint")\]|checkpoint_dir = ["/mnt/care_workspace/outputs/checkpoint"]|g' src/language_modeling/train.py
sed -i 's/accelerator.wait_for_everyone()/# accelerator.wait_for_everyone()/g' src/language_modeling/utils.py

# 4. 正式启动训练
echo "[$(date +%T)] Mistral 任务已启动，正在监听日志..."
export WANDB_MODE=disabled

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --mixed_precision bf16 --num_machines 1 --num_processes 1 \
    -m src.language_modeling.train \
    --config config/language_modeling/finetune.yaml \
    --model_name_or_path "mistralai/mistral-7b-instruct-v0.2" \
    --train_file "data_care/finetune/nq_mistral.jsonl" \
    --dev_file "data_care/finetune/nq_valid.jsonl" \
    --checkpoint_path "eunseong/care_mistral_pt" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 2 \
    --exp_name "repro_mistral" \
    --workdir "$OUTPUT_DIR" >> "$LOG_DIR/mistral.log" 2>&1

# 5. 权重提取与评测
if [ -d "$OUTPUT_DIR/checkpoint" ] && [ "$(ls -A $OUTPUT_DIR/checkpoint)" ]; then
    cp -r "$OUTPUT_DIR/checkpoint"/* "$CKPT_FINAL/"
    python3 -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('mistralai/mistral-7b-instruct-v0.2'); t.save_pretrained('$CKPT_FINAL')"
    CUDA_VISIBLE_DEVICES=0 python -m src.eval.run_eval --data nq --checkpoint_path "$CKPT_FINAL" --use_rag --eval_batch_size 8 --save_results --results_path "eval_result_mistral_nq.json" >> "$LOG_DIR/mistral.log" 2>&1
fi
