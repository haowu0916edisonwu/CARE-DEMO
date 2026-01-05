#!/bin/bash
# ---------------------------------------------------------
# CARE 复现：纯净裸模型 (Naked Mistral) 闭卷版
# ---------------------------------------------------------

PROJECT_ROOT="/mnt/care_workspace/CARE"
PHYSICAL_BASE="/mnt/care_workspace/hf_cache/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a"
LOG_FILE="${PROJECT_ROOT}/logs/mistral_naked_cb.log"
RESULTS_DIR="${PROJECT_ROOT}/eval_results"

mkdir -p "${PROJECT_ROOT}/logs" "$RESULTS_DIR"
cd "$PROJECT_ROOT" || exit

# 异步启动
nohup bash -c "
export HF_HOME='/mnt/care_workspace/hf_cache'
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled

for task in nq triviaqa webqa factkg truthfulqa; do
    echo \">> [\$(date +%T)] 正在启动裸模型【闭卷】评测: \$task\" >> '$LOG_FILE'
    
    # 核心修改点：
    # 1. 删除了 --checkpoint_path (彻底消除 LoRA 权重警告)
    # 2. 删除了 --use_rag (切换为闭卷模式)
    # 3. 结果保存为 _naked_cb.json
    python -m src.eval.run_eval_mistral \\
        --data \"\$task\" \\
        --model_name_or_path '$PHYSICAL_BASE' \\
        --eval_batch_size 8 --save_results \\
        --results_path '${RESULTS_DIR}/mistral_\${task}_naked_cb.json' \\
        >> '$LOG_FILE' 2>&1
    
    sleep 2
done
" > /dev/null 2>&1 &

echo "🚀 Mistral 裸模型闭卷评测已启动！"
echo "监控进度：tail -f $LOG_FILE"