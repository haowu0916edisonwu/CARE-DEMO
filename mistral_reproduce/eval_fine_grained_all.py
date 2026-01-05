import json
import os
import pandas as pd

# 1. 配置路径
# 根据你的描述，所有结果文件都在这个目录下
BASE_DIR = "/mnt/care_workspace/CARE/mistral_reproduce"
TASKS = ["nq", "triviaqa", "webqa", "factkg", "truthfulqa"]

def is_correct(pred, answers):
    """基础子串匹配逻辑 (Span EM)"""
    if pred is None: return False
    pred = str(pred).lower()
    return any(str(ans).lower() in pred for ans in answers)

def calculate_metrics(task):
    # 构造文件名：裸模型闭卷 vs 最终 RAG 版
    cb_path = os.path.join(BASE_DIR, f"mistral_{task}_naked_cb.json")
    rag_path = os.path.join(BASE_DIR, f"mistral_{task}_final.json")

    if not os.path.exists(cb_path) or not os.path.exists(rag_path):
        return None

    with open(cb_path, 'r') as f: cb_data = json.load(f)
    with open(rag_path, 'r') as f: rag_data = json.load(f)

    # 用 ID 匹配数据
    cb_dict = {str(item['id']): item for item in cb_data}
    
    res_total, res_correct = 0, 0
    boost_total, boost_correct = 0, 0
    total_rag_correct = 0
    matched_count = 0

    for rag in rag_data:
        idx = str(rag['id'])
        if idx not in cb_dict: continue
        
        matched_count += 1
        ans = rag['answer']
        
        cb_res = is_correct(cb_dict[idx]['pred'], ans)
        rag_res = is_correct(rag['pred'], ans)
        
        if rag_res: total_rag_correct += 1
        
        if cb_res: # Resilience: 闭卷对，RAG 也要对
            res_total += 1
            if rag_res: res_correct += 1
        else:      # Boost: 闭卷错，RAG 变对
            boost_total += 1
            if rag_res: boost_correct += 1
            
    return {
        "Dataset": task.upper(),
        "Overall EM": round(total_rag_correct / matched_count, 4) if matched_count > 0 else 0,
        "Resilience": round(res_correct / res_total, 4) if res_total > 0 else 0,
        "Boost": round(boost_correct / boost_total, 4) if boost_total > 0 else 0,
        "Res_Samples": res_total,
        "Boost_Samples": boost_total,
        "Total": matched_count
    }

# 执行统计
all_results = []
for task in TASKS:
    result = calculate_metrics(task)
    if result:
        all_results.append(result)

# 打印美化表格
if all_results:
    df = pd.DataFrame(all_results)
    print("\n" + "="*80)
    print(f"{'CARE Mistral Fine-grained Evaluation (Reproduced)':^80}")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # 保存结果
    output_csv = os.path.join(BASE_DIR, "fine_grained_table.csv")
    df.to_csv(output_csv, index=False)
    print(f"📊 统计表格已保存至: {output_csv}")
else:
    print("❌ 未能在指定目录下找到对应的 JSON 文件，请检查路径。")