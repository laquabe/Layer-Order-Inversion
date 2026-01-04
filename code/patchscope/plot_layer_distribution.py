# -*- coding: utf-8 -*-
"""
统计实体在所有层（非最早层）的出现分布
按 (hop_count, hop_idx) 作为 group
"""

import json
from collections import defaultdict
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import re


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

def normalize_text(s):
    if s is None:
        return ""
    s = str(s).lower()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", "", s)


def entity_matches(gen, entity_list):
    gen_tokens = normalize_text(gen).split()
    for e in entity_list:
        e_tokens = normalize_text(e).split()
        if len(e_tokens) == 0:
            continue
        if all(t in gen_tokens for t in e_tokens):
            return True
    return False


def load_json(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def build_lookup(json_data):
    mapping = {}
    for case in json_data:
        cid = case["case_id"]
        e1 = case["orig"]["triples_labeled"][0][0]
        hops = [(h["answer"], h.get("answer_alias", [])) for h in case["single_hops"]]
        qs = case["questions"]
        for q in qs:
            mapping[(cid, q)] = dict(e1=e1, hops=hops)
    return mapping


def plot_layer_distribution_all(csvs, json_file, out_dir="layer_plots_all", plot_mode="new"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("Loading JSON ...")
    json_data = load_json(json_file)
    lookup = build_lookup(json_data)

    print("Loading CSVs ...")
    df = pd.concat([safe_read_file(c) for c in csvs], ignore_index=True)
    df["id"] = df["id"].astype(int)

    # group layers
    layer_groups = defaultdict(list)

    print("Building distribution ...")

    for (cid, prompt), g in df.groupby(["id", "source_prompt"]):

        key = (cid, prompt)
        if key not in lookup:
            continue

        info = lookup[key]
        hop_count = len(info["hops"])

        gens = g["generation"].tolist()
        layers = g["source_layer"].tolist()

        # --- hop_idx = 0 (e1_label)
        entity_list = [info["e1"]]
        for gen, l in zip(gens, layers):
            if entity_matches(gen, entity_list):
                layer_groups[(hop_count, 0)].append(l)

        # --- hop_idx >= 1
        for hop_idx, (ent, aliases) in enumerate(info["hops"], start=1):
            entity_list = [ent, *aliases]
            for gen, l in zip(gens, layers):
                if entity_matches(gen, entity_list):
                    layer_groups[(hop_count, hop_idx)].append(l)

    # ---- 绘图 ----
    # for key, layer_list in layer_groups.items():
    #     hop_count, hop_idx = key
    #     if len(layer_list) == 0:
    #         continue

    #     plt.figure(figsize=(10, 6))
    #     sns.histplot(layer_list, kde=True, bins=20, color="steelblue", alpha=0.6)
    #     plt.title(f"All Occurrences Layer Dist (hop_count={hop_count}, hop_idx={hop_idx})")
    #     plt.xlabel("source_layer")
    #     plt.ylabel("Frequency")

    #     plt.tight_layout()
    #     save_path = Path(out_dir) / f"layer_all_{hop_count}_{hop_idx}.png"
    #     plt.savefig(save_path, dpi=150)
    #     plt.close()

    #     print(f"Saved: {save_path}")
    
        # ---- 绘图 ----
    if plot_mode == "old":
        # === 保持旧画法，每个 (hop_count, hop_idx) 一张图 ===
        for key, layer_list in layer_groups.items():
            hop_count, hop_idx = key
            if len(layer_list) == 0:
                continue

            plt.figure(figsize=(10, 6))
            sns.histplot(layer_list, kde=True, bins=20, color="steelblue", alpha=0.6)
            plt.title(f"All Occurrences Layer Dist (hop_count={hop_count}, hop_idx={hop_idx})")
            plt.xlabel("source_layer")
            plt.ylabel("Frequency")

            plt.tight_layout()
            save_path = Path(out_dir) / f"layer_all_{hop_count}_{hop_idx}.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"Saved: {save_path}")

    else:
        # === 新画法：按 hop_count 合并不同 hop_idx 到一张图 ===

        # hop_count → {hop_idx: [layers]}
        merged = defaultdict(lambda: defaultdict(list))
        for (hop_count, hop_idx), layer_list in layer_groups.items():
            merged[hop_count][hop_idx] = layer_list

        for hop_count, hop_dict in merged.items():
            plt.figure(figsize=(10, 6))

            for hop_idx, layer_list in sorted(hop_dict.items()):
                if len(layer_list) == 0:
                    continue
                
                # 计算每个 layer 的频率，准备折线图
                series = pd.Series(layer_list).value_counts().sort_index()
                plt.plot(series.index, series.values, marker="o", label=f"hop_idx={hop_idx}")

            plt.title(f"Layer Distribution (All Occurrences)\n hop_count={hop_count}")
            plt.xlabel("source_layer")
            plt.ylabel("Frequency")
            plt.legend(title="hop_idx")
            plt.tight_layout()

            save_path = Path(out_dir) / f"layer_all_{hop_count}_merged.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"Saved: {save_path}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--json", required=True, help="原 JSON 文件")
    parser.add_argument("--csvs", nargs="+", required=True, help="多个生成 CSV")
    parser.add_argument("--out_dir", default="layer_plots_all")
    parser.add_argument("--plot_mode", default="new", choices=["new", "old"], help="绘图模式：new（按 hop_count 合并），old（每个 hop_count-hop_idx 一张图）")

    args = parser.parse_args()
    plot_layer_distribution_all(args.csvs, args.json, args.out_dir, args.plot_mode)

