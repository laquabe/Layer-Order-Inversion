# -*- coding: utf-8 -*-
"""
analyze_generation_results_v4.py
新增功能：
1. 过滤低层 (source_layer < min_valid_layer) 的出现，不计入 first_layer
2. 但仍保留原始生成结果以供分析
3. case 级别为 (id + source_prompt)
"""

import json
import pandas as pd
import argparse
import numpy as np
import csv
from io import StringIO
import re
import jsonlines
import os

def safe_read_csv(path):
    print(f"\n📂 Reading CSV file: {path}")

    df = pd.read_csv(
        path,
        engine="python",
        sep=",",
        quotechar='"',
        escapechar="\\",
        doublequote=True,
        index_col=0
    )

    if df.columns[0].lower().startswith("unnamed") or df.columns[0] == "":
        print("🧹 Removing redundant index column")
        df = df.drop(df.columns[0], axis=1)

    # fix types
    for col in ["id", "source_layer", "target_layer"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            null_mask = df[col].isna()
            if null_mask.sum() > 0:
                print(f"⚠️ Removing {null_mask.sum()} invalid rows for {col}")
                df = df.loc[~null_mask]

    print(df[["id", "source_layer", "target_layer"]].dtypes)
    print(f"✅ Loaded {len(df)} rows from {path}")
    return df

def safe_read_file(path):
    """
    通用加载函数：兼容 CSV 和 JSONL
    - CSV → pandas DataFrame（老逻辑）
    - JSONL → pandas DataFrame（逐行读取）
    """
    print(f"\n📂 Reading file: {path}")

    # 判断扩展名
    lower = path.lower()

    if lower.endswith(".csv"):
        return _read_csv_safe(path)

    elif lower.endswith(".jsonl"):
        return _read_jsonl_safe(path)

    elif lower.endswith(".json"):
        # 如果 json 中是 list[dict]
        data = json.load(open(path, "r", encoding="utf-8"))
        if isinstance(data, dict):
            data = [data]
        df = pd.DataFrame(data)
        print(f"📄 Loaded {len(df)} rows from JSON")
        return df

    else:
        raise ValueError(f"❌ Unsupported file format: {path}")

def _read_csv_safe(path):
    """
    Safely load a CSV file by:
      1. Removing NULL bytes
      2. Dropping overly long or malformed lines
      3. Trying C engine first, then falling back to Python engine
    """

    # --- Step 1: Remove NULL bytes (Python engine cannot handle them)
    cleaned_path = path + ".cleaned_tmp"

    with open(path, "rb") as f_in, open(cleaned_path, "wb") as f_out:
        for line in f_in:
            # Remove NULL bytes
            cleaned = line.replace(b"\x00", b"")
            f_out.write(cleaned)

    # --- Step 2: attempt to load using C engine
    try:
        df = pd.read_csv(
            cleaned_path,
            engine="c",
            sep=",",
            quotechar='"',
            escapechar="\\",
            doublequote=True,
            index_col=0,
            on_bad_lines="skip",      # skip malformed rows
            low_memory=False,
        )
        print("Using C engine")
    except Exception as e:
        print(f"⚠️ C engine failed: {e}")
        print("➡️ switching to Python engine...")

        # --- Step 3: Python engine fallback
        df = pd.read_csv(
            cleaned_path,
            engine="python",
            sep=",",
            quotechar='"',
            escapechar="\\",
            doublequote=True,
            index_col=0,
            on_bad_lines="skip",      # skip malformed rows
        )
        print("Using Python engine")

    # --- Step 4: Remove redundant index column
    if df.columns[0].lower().startswith("unnamed") or df.columns[0] == "":
        print("🧹 Removing redundant index column")
        df = df.drop(df.columns[0], axis=1)

    # --- Step 5: Fix types
    for col in ["id", "source_layer", "target_layer"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            null_mask = df[col].isna()
            if null_mask.sum() > 0:
                print(f"⚠️ Removing {null_mask.sum()} invalid rows for {col}")
                df = df.loc[~null_mask]

    print(df[["id", "source_layer", "target_layer"]].dtypes)
    print(f"✅ Loaded {len(df)} CSV rows")

    return df

# def _read_csv_safe(path):
#     """你原来的 CSV 加载逻辑 preserved"""
#     df = pd.read_csv(
#         path,
#         engine="python",
#         sep=",",
#         quotechar='"',
#         escapechar="\\",
#         doublequote=True,
#         index_col=0
#     )

#     if df.columns[0].lower().startswith("unnamed") or df.columns[0] == "":
#         print("🧹 Removing redundant index column")
#         df = df.drop(df.columns[0], axis=1)

#     # fix types
#     for col in ["id", "source_layer", "target_layer"]:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
#             null_mask = df[col].isna()
#             if null_mask.sum() > 0:
#                 print(f"⚠️ Removing {null_mask.sum()} invalid rows for {col}")
#                 df = df.loc[~null_mask]

#     print(df[["id", "source_layer", "target_layer"]].dtypes)
#     print(f"✅ Loaded {len(df)} CSV rows")
#     return df


def _read_jsonl_safe(path):
    """JSONL → DataFrame"""
    import jsonlines
    rows = []

    with jsonlines.open(path, "r") as reader:
        for obj in reader:
            rows.append(obj)

    df = pd.DataFrame(rows)
    print(f"📄 Loaded {len(df)} rows from JSONL")
    
    # 保持和 CSV 一样的 id 类型约束
    for col in ["id", "source_layer", "target_layer"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


def load_json(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def save_to_jsonl(output_path, results):
    # 1. 创建目录（如果不存在）
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 2. 写入 jsonl 文件
    with jsonlines.open(output_path, mode='w') as writer:
        for r in results:
            for key, v in r.items():
                if isinstance(v, np.integer):
                    r[key] = int(v)
            writer.write(r)


def normalize_text(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", "", s)
    return s.strip()


def entity_matches(text, entity_list):
    """精确匹配完整词"""
    t = normalize_text(text).split()
    clean_entities = [e for e in entity_list if len(e) > 2]

    for e in clean_entities:
        e_norm = normalize_text(e).split()
        if all(w in t for w in e_norm):
            return True
    return False


def build_case_lookup(json_data):
    mapping = {}
    for case in json_data:
        cid = int(case["case_id"])
        e1_label = case["orig"]["triples_labeled"][0][0]
        single_hops = case.get("single_hops", [])
        questions = case.get("questions", [])

        for q in questions:
            mapping[(cid, q)] = {
                "e1_label": e1_label,
                "hops": [(h["answer"], h.get("answer_alias", [])) for h in single_hops]
            }
    return mapping


def analyze_entity_occurrences(json_data, csv_dfs, output_jsonl, min_valid_layer=5):
    """
    min_valid_layer: 过滤掉 source_layer < min_valid_layer 的匹配
    """
    df = pd.concat(csv_dfs, ignore_index=True)
    df["id"] = df["id"].astype(int)

    lookup = build_case_lookup(json_data)
    results = []

    print(f"\n🔍 Using min_valid_layer = {min_valid_layer}")

    # -------- 遍历 case -------
    for (cid, prompt), group in df.groupby(["id", "source_prompt"]):
        key = (cid, prompt)
        if key not in lookup:
            print(f"⚠️ Missing in JSON: id={cid}, prompt={prompt[:30]}...")
            continue

        info = lookup[key]
        e1_label = info["e1_label"]
        hops = info["hops"]
        hop_count = len(hops)

        gens = group["generation"].tolist()
        layers = group["source_layer"].tolist()

        # ====== helper 函数 ======
        def process_entity(entity, aliases, hop_idx):
            entity_list = [entity, *aliases]

            # 全层匹配（用于判断出现但低层太低）
            raw_hits = [
                (l, g) for g, l in zip(gens, layers)
                if entity_matches(g, entity_list)
            ]

            # 过滤后的匹配
            filtered_hits = [
                (l, g) for (l, g) in raw_hits
                if l >= min_valid_layer
            ]

            if filtered_hits:
                # 取最早出现层
                first_layer, first_gen = min(filtered_hits, key=lambda x: x[0])
                return {
                    "case_id": cid,
                    "source_prompt": prompt,
                    "hop_idx": hop_idx,
                    "hop_count": hop_count,
                    "entity": entity,
                    "appear_ratio": 1.0,
                    "first_layer": first_layer,
                    "generation": first_gen
                }
            else:
                # 有 raw_hit 但都 < min_valid_layer ⇒ 不计入出现
                return {
                    "case_id": cid,
                    "source_prompt": prompt,
                    "hop_idx": hop_idx,
                    "hop_count": hop_count,
                    "entity": entity,
                    "appear_ratio": 0.0,
                    "first_layer": None,
                    "generation": gens[-1]  # 保留生成
                }

        # -------- e1_label 作为 hop_idx=0 --------
        result_e1 = process_entity(e1_label, [], hop_idx=0)
        results.append(result_e1)

        # -------- 每跳实体 hop_idx=1..N --------
        for hop_idx, (entity, aliases) in enumerate(hops, start=1):
            results.append(process_entity(entity, aliases, hop_idx))

    # 保存
    save_to_jsonl(output_jsonl, results)
    print(f"✅ Results saved to {output_jsonl}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_layer", type=int, default=5)
    args = parser.parse_args()

    json_data = load_json(args.json)
    csv_dfs = [safe_read_file(c) for c in args.input]

    analyze_entity_occurrences(
        json_data,
        csv_dfs,
        output_jsonl=args.output,
        min_valid_layer=args.min_layer
    )


if __name__ == "__main__":
    main()
