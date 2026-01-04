# -*- coding: utf-8 -*-
"""
analyze_similarity.py  (GPT-J, same-layer-only, normalized, feature extraction version)
--------------------------------------------------------
特性：
1. 仅比较同层相似度（不同层映射空间不可比）
2. 每条样本先做 Z-score 归一化，消除样本间差异
3. 仅保留 last_subject 和 last 相关锚点（4种组合）
4. 输出统计特征（mean/std/min/max/range）
"""

import os
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ===================== Hook 注册（GPT-J） =====================
def register_hooks_gptj(model, target_module: str):
    buffers = {"rep": {}}
    handles = []
    for i, block in enumerate(model.transformer.h):
        def make_hook(ii):
            def hook(module, inputs, output):
                buffers["rep"][ii] = output.detach().to("cpu")
            return hook

        if target_module == "mlp.fc_in":
            handles.append(block.mlp.fc_in.register_forward_hook(make_hook(i)))
        elif target_module == "mlp.fc_out":
            handles.append(block.mlp.fc_out.register_forward_hook(make_hook(i)))
        elif target_module == "attn.out_proj":
            handles.append(block.attn.out_proj.register_forward_hook(make_hook(i)))
        else:
            raise ValueError(f"Unsupported target_module: {target_module}")
    return buffers, handles

def register_hooks_llama(model, target_module: str):
    """
    Register forward hooks for LLaMA model.
    Supported target_module:
        - "mlp.fc_in"   → LLaMA: mlp.up_proj
        - "mlp.fc_out"  → LLaMA: mlp.down_proj
        - "attn.out_proj" → LLaMA: self_attn.o_proj
    """
    buffers = {"rep": {}}
    handles = []

    # LLaMA layers are in model.model.layers
    layers = model.model.layers

    for i, block in enumerate(layers):

        def make_hook(ii):
            def hook(module, inp, out):
                buffers["rep"][ii] = out.detach().to("cpu")
            return hook

        if target_module == "mlp.fc_in":
            handles.append(block.mlp.up_proj.register_forward_hook(make_hook(i)))

        elif target_module == "mlp.fc_out":
            handles.append(block.mlp.down_proj.register_forward_hook(make_hook(i)))

        elif target_module == "attn.out_proj":
            handles.append(block.self_attn.o_proj.register_forward_hook(make_hook(i)))

        else:
            raise ValueError(f"Unsupported target_module for LLaMA: {target_module}")

    return buffers, handles

def get_model_type(model):
    mt = model.config.model_type.lower()
    if "gptj" in mt:
        return "gptj"
    elif "llama" in mt:
        return "llama"
    else:
        raise ValueError(f"Unsupported model type: {mt}")
    
# ===================== 编码 + 抽取层表示 =====================
def encode_and_run(model, tokenizer, text: str, target_module: str, device: str, max_len: int):
    
    model_type = get_model_type(model)
    if model_type == "gptj":
        buffers, handles = register_hooks_gptj(model, target_module)
    elif model_type == "llama":
        buffers, handles = register_hooks_llama(model, target_module)
    else:
        raise ValueError(f"Unsupported model type for encode_and_run: {model_type}")

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    offsets = enc["offset_mapping"][0].tolist()

    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)

    layers = sorted(buffers["rep"].keys())
    reps = [buffers["rep"][i] for i in layers]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    for h in handles:
        h.remove()

    return reps, tokens, offsets, enc["input_ids"]


# ===================== 主语提取与定位 =====================
def extract_subject_from_case(case: dict, hop_idx: int, is_multi: bool) -> Optional[str]:
    triples = case.get("orig", {}).get("triples_labeled", [])
    if not triples:
        return None
    if is_multi:
        return triples[0][0]
    if 0 <= hop_idx < len(triples):
        return triples[hop_idx][0]
    return triples[-1][0]


# def locate_subject_tokens(tokens, offsets, text, subject_name: Optional[str]) -> List[int]:
#     if not subject_name:
#         return []
#     def find_span(pattern: str):
#         m = re.search(re.escape(pattern), text, flags=re.IGNORECASE)
#         return m.span() if m else None

#     spans = []
#     full_span = find_span(subject_name.strip())
#     if full_span:
#         spans.append(full_span)
#     else:
#         for w in subject_name.strip().split():
#             sp = find_span(w)
#             if sp:
#                 spans.append(sp)

#     if not spans:
#         return []

#     idxs = set()
#     for (start, end) in spans:
#         for i, (ts, te) in enumerate(offsets):
#             if te <= ts:
#                 continue
#             if not (te <= start or ts >= end):
#                 idxs.add(i)
#     return sorted(idxs)

def locate_subject_tokens(model_type, tokenizer, tokens, offsets, text, subject_name, subject_input_ids):
    if not subject_name:
        return []

    # ======================================================
    # GPT-J: 依旧走 offset 匹配
    # ======================================================
    if model_type == "gptj":
        spans = []
        subj = subject_name.strip()

        # full match
        m = re.search(re.escape(subj), text, flags=re.IGNORECASE)
        if m:
            spans.append(m.span())
        else:
            for w in subj.split():
                m = re.search(re.escape(w), text, flags=re.IGNORECASE)
                if m:
                    spans.append(m.span())

        if not spans:
            return []

        idxs = set()
        for (start, end) in spans:
            for i, (ts, te) in enumerate(offsets):
                if te <= ts:
                    continue
                if not (te <= start or ts >= end):
                    idxs.add(i)
        return sorted(idxs)

    # ======================================================
    # LLaMA: Token-based 匹配（含 自动加空格）
    # ======================================================
    elif model_type == "llama":

        def find_tokens_llama(substring, prepend_space=False, last=True):
            """LLaMA token 匹配，支持加空格与不加空格"""
            if prepend_space:
                substring = " " + substring

            st = tokenizer(substring, add_special_tokens=False, return_tensors="pt").input_ids[0]
            st = st.to(subject_input_ids.device)

            full = subject_input_ids[0]
            L, K = len(full), len(st)

            for start in range(L - K + 1):
                end = start + K
                if torch.all(full[start:end] == st):
                    return [end - 1] if last else list(range(start, end))
            return None

        name = subject_name.strip()

        # 尝试 2：自动加空格匹配 SentencePiece 的 "▁word"
        idxs = find_tokens_llama(name, prepend_space=True, last=True)
        if idxs:
            return idxs

    else:
        raise ValueError(f"Unsupported model type: {model_type}")



# def get_anchor_indices(tokens, offsets, text, subject_name: Optional[str]) -> Dict[str, Optional[List[int]]]:
#     subj_idxs = locate_subject_tokens(tokens, offsets, text, subject_name)
#     n = len(tokens)
#     anchors = {
#         "last_subject": [subj_idxs[-1]] if len(subj_idxs) >= 1 else None,
#         "last": [n - 1] if n > 0 else None,
#     }
#     return anchors

def get_anchor_indices(model_type, tokenizer, tokens, offsets, text, subject_name, input_ids):
    subj_idxs = locate_subject_tokens(
        model_type=model_type,
        tokenizer=tokenizer,
        tokens=tokens,
        offsets=offsets,
        text=text,
        subject_name=subject_name,
        subject_input_ids=input_ids,
    )
    if subj_idxs is None or len(subj_idxs) == 0:
        return None

    n = len(tokens)
    anchors = {
        "last_subject": [subj_idxs[-1]] if len(subj_idxs) > 0 else None,
        "last": [n - 1] if n > 0 else None,
    }
    return anchors

# ===================== Cross-anchor 同层相似度 + 句间基线归一化 =====================
def layer_layer_similarity_on_anchors(
    reps_multi: List[torch.Tensor],
    reps_single: List[torch.Tensor],
    anchors_multi: Dict[str, Optional[List[int]]],
    anchors_single: Dict[str, Optional[List[int]]],
) -> Tuple[Dict[str, np.ndarray], List[Tuple[str, str]]]:
    """
    计算多跳和单跳在同层上的关键token相似度，并做句间级baseline归一化。
    返回：
        matrices: { "last_subject__last": np.array([layer0, layer1, ...]) }
        pair_names: [(km, ks), ...]
    """

    L = min(len(reps_multi), len(reps_single))  # 层数对齐
    matrices = {}

    # -------- 获取有效锚点 --------
    multi_keys = [k for k, v in anchors_multi.items() if v]
    single_keys = [k for k, v in anchors_single.items() if v]
    valid_pairs = [(km, ks) for km in multi_keys for ks in single_keys]

    def mean_vec(X: torch.Tensor, idxs: List[int]) -> Optional[torch.Tensor]:
        """取若干token的平均表示"""
        if not idxs:
            return None
        return X[idxs].mean(dim=0) if len(idxs) > 1 else X[idxs[0]]

    def sentence_baseline(Xm: torch.Tensor, Xs: torch.Tensor) -> float:
        """整句层级的baseline：用平均向量的cosine相似度"""
        vm = Xm.mean(dim=0)
        vs = Xs.mean(dim=0)
        return F.cosine_similarity(vm.float(), vs.float(), dim=-1).item()

    # -------- 逐层、逐锚点对计算相似度 --------
    for (km, ks) in valid_pairs:
        sims = []
        for l in range(L):
            Xm = reps_multi[l][0]  # [seq_len, hidden_dim]
            Xs = reps_single[l][0]

            vm = mean_vec(Xm, anchors_multi[km])
            vs = mean_vec(Xs, anchors_single[ks])

            if vm is None or vs is None:
                sims.append(0.0)
                continue

            # 计算token相似度
            sim = F.cosine_similarity(vm.float(), vs.float(), dim=-1).item()

            # 计算baseline并归一化
            baseline = sentence_baseline(Xm, Xs)
            sims.append(sim - baseline)

        matrices[f"{km}__{ks}"] = np.array(sims, dtype=np.float32)

    return matrices

# ===================== 归一化函数 =====================
def normalize_matrix(A: np.ndarray, method="zscore"):
    if method == "zscore":
        mean, std = A.mean(), A.std()
        return (A - mean) / std if std > 1e-6 else A - mean
    elif method == "minmax":
        A_min, A_max = A.min(), A.max()
        return (A - A_min) / (A_max - A_min + 1e-6)
    else:
        return A


# ===================== 绘制同层曲线 =====================
def plot_same_layer_curve(diag_vec: np.ndarray, save_path: Path, title: str):
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(diag_vec)), diag_vec, marker="o")
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel("Similarity (same layer)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

# ===================== 主流程 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="EleutherAI/gpt-j-6B")
    parser.add_argument("--target_module", type=str, default="mlp.fc_in",
                        choices=["mlp.fc_in", "mlp.fc_out", "attn.out_proj"])
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--out_dir", type=str, default="outputs_v2")
    parser.add_argument("--cpu_threads", type=int, default=10)
    parser.add_argument("--norm", type=str, default="zscore", choices=["zscore", "minmax", "none"])
    args = parser.parse_args()

    # 限制 CPU 使用
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    torch.set_num_threads(args.cpu_threads)
    print(f"[CPU限制] 线程数上限: {args.cpu_threads}")

    # 加载模型
    print(f"[加载模型] {args.model}，目标模块: {args.target_module}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    model.eval()
    print(f"[模型加载完成] 使用设备: {device}")

    # 数据加载
    print(f"[加载数据] {args.data}")
    cases = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if isinstance(cases, dict):
        cases = [cases]

    out_root = Path(args.out_dir)
    combined_dir = out_root / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    combined: Dict[str, Dict[str, List[np.ndarray]]] = {}

    # 核心循环
    for case in tqdm(cases):
        cid = case.get("case_id", "unknown")
        multis = case["questions"]
        singles = [sh["question"] for sh in case["single_hops"]]
        y = len(singles)

        multi_subject = extract_subject_from_case(case, 0, is_multi=True)
        reps_multi_all, anchors_multi_all = [], []
        for q_m in multis:
            reps_m, tokens_m, offsets_m, input_ids_m= encode_and_run(model, tokenizer, q_m, args.target_module, device, args.max_len)
            anchors_m = get_anchor_indices(
                model_type=get_model_type(model),
                tokenizer=tokenizer,
                tokens=tokens_m,
                offsets=offsets_m,
                text=q_m,
                subject_name=multi_subject,
                input_ids=input_ids_m,  # 注意要把 enc 返回出来
            )
            if anchors_m is None:
                print(f"[Skip] Multi-hop subject not found for case {cid}")
                continue
            reps_multi_all.append(reps_m)
            anchors_multi_all.append(anchors_m)

        for x, q_s in enumerate(singles, start=1):
            single_subject = extract_subject_from_case(case, x - 1, is_multi=False)
            reps_s, tokens_s, offsets_s, input_ids_s = encode_and_run(model, tokenizer, q_s, args.target_module, device, args.max_len)
            anchors_s = get_anchor_indices(
                model_type=get_model_type(model),
                tokenizer=tokenizer,
                tokens=tokens_s,
                offsets=offsets_s,
                text=q_s,
                subject_name=single_subject,
                input_ids=input_ids_s,  # 注意要把 enc 返回出来
            )
            if anchors_s is None:
                print(f"[Skip] Single-hop subject not found for case {cid}, q={q_s}")
                continue

            pair_vecs_list, pair_names_master = [], None
            for reps_m, anchors_m in zip(reps_multi_all, anchors_multi_all):
                mats_dict = layer_layer_similarity_on_anchors(reps_m, reps_s, anchors_m, anchors_s)
                if pair_names_master is None:
                    pair_names_master = list(mats_dict.keys())
                pair_vecs_list.append(mats_dict)

            avg_vecs = {}
            for key in (pair_names_master or []):
                normed = [d[key] for d in pair_vecs_list if key in d]
                stack = np.stack(normed, axis=0)
                avg_vecs[key] = stack.mean(axis=0)

            key_bucket = f"single_{x}-of-{y}"
            combined.setdefault(key_bucket, {})
            for pair_key, v in avg_vecs.items():
                combined[key_bucket].setdefault(pair_key, []).append(v)

    # 汇总 + 统计
    records = []
    for key_bucket, pair_bucket in combined.items():
        for pair_key, vecs in pair_bucket.items():
            arr = np.stack(vecs, axis=0)
            A_global = arr.mean(axis=0)
            tag = f"{key_bucket}__{pair_key}_L{len(A_global)}"
            np.save(combined_dir / f"{tag}.npy", A_global)
            plot_same_layer_curve(A_global, combined_dir / f"{tag}.png",
                                  f"[Same-layer] {key_bucket} [{pair_key}]")

            feat = {
                "pair": pair_key,
                "case_type": key_bucket,
                "mean": float(A_global.mean()),
                "std": float(A_global.std()),
                "min": float(A_global.min()),
                "max": float(A_global.max()),
                "range": float(A_global.max() - A_global.min()),
            }
            records.append(feat)

    df = pd.DataFrame(records)
    stats_path = out_root / "feature_summary.csv"
    df.to_csv(stats_path, index=False)
    print(f"[Feature summary saved] {stats_path}")


if __name__ == "__main__":
    main()
