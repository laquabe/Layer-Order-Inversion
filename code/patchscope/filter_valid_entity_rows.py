import pandas as pd
import torch
from transformers import AutoTokenizer
from pathlib import Path

# ========= 你提供的 find_tokens =========

def find_tokens(tokenizer, string_tokens, substring, prepend_space, last=True):
    if prepend_space:
        substring = " " + substring

    substring_tokens = tokenizer(
        substring, add_special_tokens=False, return_tensors="pt"
    ).input_ids[0]

    substring_tokens = substring_tokens.to(string_tokens.device)

    L = len(substring_tokens)
    S = len(string_tokens)

    for start in range(S - L + 1):
        end = start + L
        if torch.all(string_tokens[start:end] == substring_tokens):
            return end - 1 if last else (start, end - 1)

    return None

# ============ 根据模型自动判断是否需要 prepend_space ============

def get_prepend_space(model_name):
    """
    Llama / GPT-J / GPT2 系模型 → prepend_space = True
    Qwen → False
    """
    name = model_name.lower()
    if "gpt" in name and ("j" in name or "neo" in name):
        return True
    if 'llama-3' in name:
        return True
    return False

def clean_dataframe(df):
    """
    删除 Unnamed 列 & 重建 index
    """
    # 删除 Unnamed... 列
    unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed_cols:
        print(f"🧹 Removing redundant columns: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)

    # 重建 index
    df = df.reset_index(drop=True)

    return df

# ================= 主过滤逻辑 =================

def filter_valid_entity_rows(input_csv, output_csv, model_name,
                             entity_col="e1_label", prompt_col="source_prompt", check_last_token=False):
    print(f"📂 Loading CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    df = clean_dataframe(df)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    prepend_space = get_prepend_space(model_name)

    keep_mask = []

    for idx, row in df.iterrows():
        entity = str(row[entity_col])
        prompt = str(row[prompt_col])

        # 如果 check_last_token 为 True，则检查最后一个字符是否匹配
        if check_last_token:
            last_token = prompt.split()[-1]
            tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
            token_pos = find_tokens(tokenizer, tokens, last_token, prepend_space)

        else:
            # tokenize prompt 得到 token 序列
            tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
            token_pos = find_tokens(tokenizer, tokens, entity, prepend_space)

        keep_mask.append(token_pos is not None)

        if idx % 300 == 0:
            print(f"  → processed {idx}/{len(df)}")

    df_filtered = df.loc[keep_mask].reset_index(drop=True)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_csv, index=True)

    print("\n✨ Done!")
    print(f"  原始样本: {len(df)}")
    print(f"  有效样本: {len(df_filtered)}")
    print(f"  已保存到: {output_csv}")


# ================= CLI =================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--check_last_token", action="store_true", default=False, 
                        help="Set this flag to check the last token instead of the entity label")

    args = parser.parse_args()

    filter_valid_entity_rows(
        input_csv=args.input,
        output_csv=args.output,
        model_name=args.model,
        check_last_token=args.check_last_token
    )
