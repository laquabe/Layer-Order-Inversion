import json
import pandas as pd


def convert_json_to_entitydesc_input(json_path, output_csv, take_first_only=False):
    """
    将分类后的 JSON 数据转换为 Patchscopes 实验输入格式的 CSV。
    
    参数:
        json_path (str): 输入 JSON 文件路径
        output_csv (str): 输出 CSV 文件路径
        take_first_only (bool): 若为 True，则每个 case 只取第一个 question；
                                若为 False，则展开所有 questions。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 若顶层是单个对象，包成列表
    if isinstance(data, dict):
        data = [data]

    records = []
    for item in data:
        case_id = item.get("case_id")
        subj = item["orig"]["triples_labeled"][0][0]
        r1_template = item["requested_rewrite"][0].get("prompt", "{}")

        # 获取所有 questions
        questions = item.get("questions", [])

        # 若 take_first_only=True，只取第一个
        if take_first_only and len(questions) > 0:
            questions = [questions[0]]

        for q in questions:
            # 保证 e1_label 在 prompt 中出现
            assert subj in q, f"[Data Error] e1_label '{subj}' not found in source_prompt: {q}"

            records.append({
                "id": case_id,
                "source_prompt": q,
                "e1_label": subj,
                "r1_template": r1_template
            })

    df = pd.DataFrame(records)
    # index=True => 保持 Patchscopes 兼容格式
    df.to_csv(output_csv, index=True, encoding="utf-8")
    print(f"✅ Converted {len(df)} samples into {output_csv}")
    print(df.head())


# 用法
convert_json_to_entitydesc_input(
    "/data/xkliu/Knowledge_Edit/analyze/Llama-3-8B_classified_split_results/class_3_single_correct_multi_wrong.json",
    "datasets/Llama-3-8B_classified_split_results_sample/class_3_single_correct_multi_wrong.csv",
    take_first_only=True
)
