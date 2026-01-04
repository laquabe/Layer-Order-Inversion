import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# ===== Mean Pooling (HuggingFace 官方写法) =====
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / \
           torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)


# ===== CSV 读取 =====
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


# ===== Text → Embedding =====
def encode_text(text):
    encoded = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)

    with torch.no_grad():
        model_output = model(**encoded)

    emb = mean_pooling(model_output, encoded["attention_mask"])
    emb = F.normalize(emb, p=2, dim=1)

    return emb[0].cpu().numpy()


def get_similarity(t1, t2):
    emb1 = encode_text(str(t1))
    emb2 = encode_text(str(t2))
    return np.dot(emb1, emb2)


# ====== ⭐ 核心：两种过滤模式 ⭐ ======
def filter_global(df, percentile=10):
    """整体过滤：所有样本混在一起取阈值"""
    print("\n🔎 Using GLOBAL filtering (all samples together)...")

    similarities = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sim = get_similarity(row["generation"], row["source_prompt"])
        similarities.append(sim)

    df["similarity"] = similarities

    threshold_value = np.percentile(similarities, percentile)
    print(f"🎯 Global threshold = {threshold_value:.4f}")

    return df[df["similarity"] >= threshold_value]


def filter_case(df, percentile=10):
    """
    按 case（id + source_prompt）分别过滤
    一个 case = (id, source_prompt)
    对每个 case 内部的多行（比如不同 layer 的 generation）算相似度，
    然后保留该 case 内 similarity 最高的 top (100 - percentile)%。
    """
    print("\n🔎 Using CASE filtering (group by id + source_prompt)...")

    df = df.copy()
    result_rows = []

    # 按 (id, source_prompt) 分组，而不是只按 id
    groups = df.groupby(["id", "source_prompt"], sort=False)
    total_cases = len(groups)

    for (case_id, src_prompt), case_df in tqdm(groups, total=total_cases):
        case_df = case_df.copy()

        # 计算当前 case 内每一行的相似度
        sims = []
        for _, row in case_df.iterrows():
            sim = get_similarity(row["generation"], row["source_prompt"])
            sims.append(sim)

        case_df["similarity"] = sims

        # 例如 percentile=10 → 保留该 case 内 similarity ≥ 10 分位数的行
        threshold = np.percentile(sims, percentile)
        keep_df = case_df[case_df["similarity"] >= threshold]

        result_rows.append(keep_df)

    if len(result_rows) == 0:
        # 没留下任何样本，返回空表但结构一致
        return df.iloc[0:0].copy()

    filtered_df = pd.concat(result_rows, ignore_index=True)
    return filtered_df


# ===== JSONL 保存 =====
def save_jsonl_from_df(df, output_path):
    import jsonlines
    with jsonlines.open(output_path, "w") as writer:
        for _, row in df.iterrows():
            writer.write(row.to_dict())
    print(f"💾 Saved {len(df)} rows → {output_path}")


# ===== 主流程 =====
def main(file_name, token, mode="global", percentile=10):
    input_csv = f"output/Llama-3-8B_classified_split_results_sample/description/{token}/run_0/{file_name}_valid.jsonl"
    output_jsonl = f"output/Llama-3-8B_classified_split_results_sample/description/{token}/run_0/{file_name}_{mode}Filter{percentile}.jsonl"

    df = safe_read_file(input_csv)

    # ⭐ 根据 mode 选择过滤模式 ⭐
    if mode == "global":
        filtered_df = filter_global(df, percentile=percentile)
    elif mode == "case":
        filtered_df = filter_case(df, percentile=percentile)
    else:
        raise ValueError("mode must be 'global' or 'case'")

    save_jsonl_from_df(filtered_df, output_jsonl)


# ===== 入口 =====
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "/data/xkliu/hf_models/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # ------------ 修改这里就可以切换模式 ------------
    main('class_4_both_wrong','last',mode="global", percentile=90)   # 或 main(mode="global")

    # class_1_both_correct
    # class_3_single_correct_multi_wrong
    # class_4_both_wrong
