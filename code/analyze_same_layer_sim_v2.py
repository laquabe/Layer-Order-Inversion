# -*- coding: utf-8 -*-
"""
analyze_similarity_v3.py  (GPT-J, same-layer-only, raw + baseline-normalized)
--------------------------------------------------------
改动：
1. 分组口径仍然是：single_{x}-of-{y} （总条数-当前跳数）
2. 对每个样本、每个锚点对，计算同层相似度：
   - raw:  直接 cosine(vm, vs)
   - norm: raw - sentence_baseline
3. 对每个 bucket (single_x-of-y) + pair_key：
   - 聚合所有 case，得到平均层曲线
   - 同时输出 raw / norm 的 npy + png
   - 在 CSV 中输出 raw / norm 的统计特征
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


# ===================== 编码 + 抽取层表示 =====================
def encode_and_run(model, tokenizer, text: str, target_module: str, device: str, max_len: int):
    buffers, handles = register_hooks_gptj(model, target_module)
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

    return reps, tokens, offsets


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


def locate_subject_tokens(tokens, offsets, text, subject_name: Optional[str]) -> List[int]:
    if not subject_name:
        return []

    def find_span(pattern: str):
        m = re.search(re.escape(pattern), text, flags=re.IGNORECASE)
        return m.span() if m else None

    spans = []
    full_span = find_span(subject_name.strip())
    if full_span:
        spans.append(full_span)
    else:
        for w in subject_name.strip().split():
            sp = find_span(w)
            if sp:
                spans.append(sp)

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


def get_anchor_indices(tokens, offsets, text, subject_name: Optional[str]) -> Dict[str, Optional[List[int]]]:
    subj_idxs = locate_subject_tokens(tokens, offsets, text, subject_name)
    n = len(tokens)
    anchors = {
        "last_subject": [subj_idxs[-1]] if len(subj_idxs) >= 1 else None,
        "last": [n - 1] if n > 0 else None,
    }
    return anchors


# ===================== Cross-anchor 同层相似度（raw + baseline-norm） =====================
def layer_layer_similarity_on_anchors_with_raw(
    reps_multi: List[torch.Tensor],
    reps_single: List[torch.Tensor],
    anchors_multi: Dict[str, Optional[List[int]]],
    anchors_single: Dict[str, Optional[List[int]]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[Tuple[str, str]]]:
    """
    返回：
        raw_mats:  { "last_subject__last": np.array([L]) }  # 直接 cosine
        norm_mats: { "last_subject__last": np.array([L]) }  # cosine - baseline
        pair_names: [(km, ks), ...]
    """

    L = min(len(reps_multi), len(reps_single))
    raw_mats: Dict[str, np.ndarray] = {}
    norm_mats: Dict[str, np.ndarray] = {}

    # -------- 获取有效锚点 --------
    multi_keys = [k for k, v in anchors_multi.items() if v]
    single_keys = [k for k, v in anchors_single.items() if v]
    valid_pairs = [(km, ks) for km in multi_keys for ks in single_keys]

    def mean_vec(X: torch.Tensor, idxs: List[int]) -> Optional[torch.Tensor]:
        if not idxs:
            return None
        return X[idxs].mean(dim=0) if len(idxs) > 1 else X[idxs[0]]

    def sentence_baseline(Xm: torch.Tensor, Xs: torch.Tensor) -> float:
        vm = Xm.mean(dim=0)
        vs = Xs.mean(dim=0)
        return F.cosine_similarity(vm.float(), vs.float(), dim=-1).item()

    # -------- 逐层、逐锚点对计算相似度 --------
    for (km, ks) in valid_pairs:
        sims_raw = []
        sims_norm = []
        for l in range(L):
            Xm = reps_multi[l][0]  # [seq_len, hidden_dim]
            Xs = reps_single[l][0]

            vm = mean_vec(Xm, anchors_multi[km])
            vs = mean_vec(Xs, anchors_single[ks])

            if vm is None or vs is None:
                sims_raw.append(0.0)
                sims_norm.append(0.0)
                continue

            sim_raw = F.cosine_similarity(vm.float(), vs.float(), dim=-1).item()
            baseline = sentence_baseline(Xm, Xs)
            sim_norm = sim_raw - baseline

            sims_raw.append(sim_raw)
            sims_norm.append(sim_norm)

        key = f"{km}__{ks}"
        raw_mats[key] = np.array(sims_raw, dtype=np.float32)
        norm_mats[key] = np.array(sims_norm, dtype=np.float32)

    return raw_mats, norm_mats, valid_pairs


# ===================== 绘制同层曲线 =====================
def plot_same_layer_curve(diag_vec: np.ndarray, save_path: Path, title: str, ylabel: str = "Similarity"):
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(diag_vec)), diag_vec, marker="o")
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
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
    parser.add_argument("--out_dir", type=str, default="outputs_v3")
    parser.add_argument("--cpu_threads", type=int, default=10)
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
    combined_raw_dir = out_root / "combined_raw"
    combined_norm_dir = out_root / "combined_norm"
    combined_raw_dir.mkdir(parents=True, exist_ok=True)
    combined_norm_dir.mkdir(parents=True, exist_ok=True)

    # combined_*[key_bucket][pair_key] -> List[np.ndarray]
    combined_raw: Dict[str, Dict[str, List[np.ndarray]]] = {}
    combined_norm: Dict[str, Dict[str, List[np.ndarray]]] = {}

    # 核心循环
    for case in tqdm(cases):
        cid = case.get("case_id", "unknown")
        multis = case["questions"]
        singles = [sh["question"] for sh in case["single_hops"]]
        y = len(singles)  # 总单跳数

        # 多跳主语和表示
        multi_subject = extract_subject_from_case(case, 0, is_multi=True)
        reps_multi_all, anchors_multi_all = [], []
        for q_m in multis:
            reps_m, tokens_m, offsets_m = encode_and_run(model, tokenizer, q_m, args.target_module, device, args.max_len)
            anchors_m = get_anchor_indices(tokens_m, offsets_m, q_m, multi_subject)
            reps_multi_all.append(reps_m)
            anchors_multi_all.append(anchors_m)

        # 单跳循环
        for x, q_s in enumerate(singles, start=1):
            key_bucket = f"single_{x}-of-{y}"  # 统计口径：总条数-当前跳数（沿用原逻辑）
            single_subject = extract_subject_from_case(case, x - 1, is_multi=False)
            reps_s, tokens_s, offsets_s = encode_and_run(model, tokenizer, q_s, args.target_module, device, args.max_len)
            anchors_s = get_anchor_indices(tokens_s, offsets_s, q_s, single_subject)

            for reps_m, anchors_m in zip(reps_multi_all, anchors_multi_all):
                raw_dict, norm_dict, pair_names = layer_layer_similarity_on_anchors_with_raw(
                    reps_m, reps_s, anchors_m, anchors_s
                )

                combined_raw.setdefault(key_bucket, {})
                combined_norm.setdefault(key_bucket, {})

                for pair_key in raw_dict.keys():
                    combined_raw[key_bucket].setdefault(pair_key, []).append(raw_dict[pair_key])
                    combined_norm[key_bucket].setdefault(pair_key, []).append(norm_dict[pair_key])

    # 汇总 + 统计
    records = []
    for key_bucket, pair_bucket in combined_raw.items():
        for pair_key, vecs_raw in pair_bucket.items():
            vecs_norm = combined_norm[key_bucket][pair_key]

            arr_raw = np.stack(vecs_raw, axis=0)      # [num_samples, num_layers]
            arr_norm = np.stack(vecs_norm, axis=0)    # [num_samples, num_layers]

            A_raw = arr_raw.mean(axis=0)
            A_norm = arr_norm.mean(axis=0)

            tag_base = f"{key_bucket}__{pair_key}_L{len(A_raw)}"

            # 保存 npy
            np.save(combined_raw_dir / f"{tag_base}.npy", A_raw)
            np.save(combined_norm_dir / f"{tag_base}.npy", A_norm)

            # 保存曲线图
            plot_same_layer_curve(A_raw, combined_raw_dir / f"{tag_base}.png",
                                  f"[Same-layer RAW] {key_bucket} [{pair_key}]", ylabel="Similarity (raw)")
            plot_same_layer_curve(A_norm, combined_norm_dir / f"{tag_base}.png",
                                  f"[Same-layer NORM] {key_bucket} [{pair_key}]", ylabel="Similarity (raw - baseline)")

            # 统计特征
            feat = {
                "case_type": key_bucket,
                "pair": pair_key,
                "mean_raw": float(A_raw.mean()),
                "std_raw": float(A_raw.std()),
                "min_raw": float(A_raw.min()),
                "max_raw": float(A_raw.max()),
                "range_raw": float(A_raw.max() - A_raw.min()),
                "mean_norm": float(A_norm.mean()),
                "std_norm": float(A_norm.std()),
                "min_norm": float(A_norm.min()),
                "max_norm": float(A_norm.max()),
                "range_norm": float(A_norm.max() - A_norm.min()),
            }
            records.append(feat)

    df = pd.DataFrame(records)
    stats_path = out_root / "feature_summary.csv"
    df.to_csv(stats_path, index=False)
    print(f"[Feature summary saved] {stats_path}")
    print(f"[RAW curves saved] {combined_raw_dir}")
    print(f"[NORM curves saved] {combined_norm_dir}")


if __name__ == "__main__":
    main()
