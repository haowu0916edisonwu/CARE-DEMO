#!/bin/bash
# ---------------------------------------------------------
# CARE Mistral 最终版：调用已加固的 run_eval_mistral.py
# ---------------------------------------------------------

# 1. 核心路径定义
PROJECT_ROOT="/mnt/care_workspace/CARE"
# 物理底座路径
PHYSICAL_BASE="/mnt/care_workspace/hf_cache/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a"
# 复现的检查点路径
CKPT_PATH="${PROJECT_ROOT}/mistral_reproduce/checkpoints/mistral/step_1500"
LOG_FILE="${PROJECT_ROOT}/logs/mistral_final_reproduce.log"
RESULTS_DIR="${PROJECT_ROOT}/eval_results"

mkdir -p "${PROJECT_ROOT}/logs" "$RESULTS_DIR"
cd "$PROJECT_ROOT" || exit

# 2. 配置文件对齐 (确保 run_eval_mistral 向上跳两级能找到 config.yaml)
mkdir -p "${PROJECT_ROOT}/mistral_reproduce/checkpoints"
cp "${PROJECT_ROOT}/config/language_modeling/finetune.yaml" "${PROJECT_ROOT}/mistral_reproduce/checkpoints/config.yaml"

# 3. 异步启动全量评测
# 使用 nohup 确保关闭终端后任务不中断，并使用 \$ 保护内部循环变量
nohup bash -c "
export HF_HOME='/mnt/care_workspace/hf_cache'
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled

# 依次跑五个数据集
for task in nq triviaqa webqa factkg truthfulqa; do
    echo \">> [\$(date +%T)] 正在启动评测任务: \$task\" >> '$LOG_FILE'
    
    # 注意这里改成了你新命名的模块名 run_eval_mistral
    python -m src.eval.run_eval_mistral \\
        --data \"\$task\" \\
        --model_name_or_path '$PHYSICAL_BASE' \\
        --checkpoint_path '$CKPT_PATH' \\
        --use_rag --eval_batch_size 8 --save_results \\
        --results_path '${RESULTS_DIR}/mistral_\${task}_final.json' \\
        >> '$LOG_FILE' 2>&1
    
    sleep 2
done
" > /dev/null 2>&1 &

echo "🚀 Mistral 专用版评测已启动！"
echo "监控进度请执行：tail -f $LOG_FILE"