# -*- coding: utf-8 -*-
"""
analyze_similarity.py  (GPT-J, cross-anchor, aggregated-only version)

支持：
- --cpu_threads 控制 CPU 占用（默认 10）
- --out_dir 指定输出目录
- 仅输出最终 combined 汇总结果
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM


# ===================== Hook 注册（GPT-J） =====================
def register_hooks_gptj(model, target_module: str):
    buffers = {"rep": {}}
    handles = []

    for i, block in enumerate(model.transformer.h):
        if target_module == "mlp.fc_in":
            def make_post(ii):
                def fwd_hook(module, inputs, output):
                    buffers["rep"][ii] = output.detach().to("cpu")
                return fwd_hook
            handles.append(block.mlp.fc_in.register_forward_hook(make_post(i)))
        elif target_module == "mlp.fc_out":
            def make_post(ii):
                def fwd_hook(module, inputs, output):
                    buffers["rep"][ii] = output.detach().to("cpu")
                return fwd_hook
            handles.append(block.mlp.fc_out.register_forward_hook(make_post(i)))
        elif target_module == "attn.out_proj":
            def make_post(ii):
                def fwd_hook(module, inputs, output):
                    buffers["rep"][ii] = output.detach().to("cpu")
                return fwd_hook
            handles.append(block.attn.out_proj.register_forward_hook(make_post(i)))
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


# ===================== 主语抽取 =====================
def extract_subject_from_case(case: dict, hop_idx: int, is_multi: bool) -> Optional[str]:
    triples = case.get("orig", {}).get("triples_labeled", [])
    if not triples:
        return None
    if is_multi:
        return triples[0][0]
    if 0 <= hop_idx < len(triples):
        return triples[hop_idx][0]
    return triples[-1][0]


# ===================== 主语 token 定位 =====================
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


# ===================== 锚点定义 =====================
def get_anchor_indices(tokens, offsets, text, subject_name: Optional[str]) -> Dict[str, Optional[List[int]]]:
    subj_idxs = locate_subject_tokens(tokens, offsets, text, subject_name)
    n = len(tokens)
    anchors = {
        "subject_all": subj_idxs if subj_idxs else None,
        "first_subject": [subj_idxs[0]] if len(subj_idxs) >= 1 else None,
        "last_subject": [subj_idxs[-1]] if len(subj_idxs) >= 1 else None,
        "first": [0] if n > 0 else None,
        "last": [n - 1] if n > 0 else None,
    }
    return anchors


# ===================== Cross-anchor 层层相似度 =====================
def layer_layer_similarity_on_anchors(
    reps_multi: List[torch.Tensor],
    reps_single: List[torch.Tensor],
    anchors_multi: Dict[str, Optional[List[int]]],
    anchors_single: Dict[str, Optional[List[int]]],
) -> Tuple[Dict[str, np.ndarray], List[Tuple[str, str]]]:
    Lm, Ls = len(reps_multi), len(reps_single)

    def mean_vec(X: torch.Tensor, idxs: List[int]) -> Optional[torch.Tensor]:
        if not idxs:
            return None
        return X[idxs].mean(dim=0) if len(idxs) > 1 else X[idxs[0]]

    multi_keys = [k for k, v in anchors_multi.items() if v]
    single_keys = [k for k, v in anchors_single.items() if v]
    pair_names = [(km, ks) for km in multi_keys for ks in single_keys]

    matrices = {f"{km}__{ks}": np.zeros((Lm, Ls), dtype=np.float32) for (km, ks) in pair_names}

    for lm in range(Lm):
        Xm = reps_multi[lm][0]
        for ls in range(Ls):
            Xs = reps_single[ls][0]
            for (km, ks) in pair_names:
                im_list = anchors_multi[km]
                is_list = anchors_single[ks]
                vm = mean_vec(Xm, im_list)
                vs = mean_vec(Xs, is_list)
                sim_val = F.cosine_similarity(vm.float(), vs.float(), dim=-1).item() if vm is not None and vs is not None else 0.0
                matrices[f"{km}__{ks}"][lm, ls] = sim_val

    return matrices, pair_names


# ===================== 绘图 =====================
def plot_layer_layer_matrix(A: np.ndarray, save_path: Path, title: str, cmap: str = "viridis"):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(A, aspect="auto", interpolation="nearest", vmin=-1, vmax=1, cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("Single-hop layers")
    plt.ylabel("Multi-hop layers")
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
    parser.add_argument("--cmap", type=str, default="viridis")
    parser.add_argument("--out_dir", type=str, default="outputs",
                        help="输出目录（默认 ./outputs）")
    parser.add_argument("--cpu_threads", type=int, default=10,
                        help="限制 CPU 使用线程数（默认 10）")
    args = parser.parse_args()

    # ================= CPU 线程限制 =================
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    torch.set_num_threads(args.cpu_threads)
    print(f"[CPU限制] 线程数上限: {args.cpu_threads}")

    # ================= 模型准备 =================
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

    # ================= 数据加载 =================
    cases = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if isinstance(cases, dict):
        cases = [cases]

    out_root = Path(args.out_dir)
    combined_dir = out_root / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    combined: Dict[str, Dict[str, List[np.ndarray]]] = {}

    # ================= 核心循环 =================
    for case in cases:
        cid = case.get("case_id", "unknown")
        print(f"\n========== Case {cid} ==========")

        multis = case["questions"]
        singles = [sh["question"] for sh in case["single_hops"]]
        y = len(singles)

        multi_subject = extract_subject_from_case(case, 0, is_multi=True)
        reps_multi_all, anchors_multi_all = [], []
        for q_m in multis:
            reps_m, tokens_m, offsets_m = encode_and_run(model, tokenizer, q_m, args.target_module, device, args.max_len)
            anchors_m = get_anchor_indices(tokens_m, offsets_m, q_m, multi_subject)
            reps_multi_all.append(reps_m)
            anchors_multi_all.append(anchors_m)

        for x, q_s in enumerate(singles, start=1):
            print(f"[Single {x}-of-{y}] {q_s}")
            single_subject = extract_subject_from_case(case, x - 1, is_multi=False)
            reps_s, tokens_s, offsets_s = encode_and_run(model, tokenizer, q_s, args.target_module, device, args.max_len)
            anchors_s = get_anchor_indices(tokens_s, offsets_s, q_s, single_subject)

            pair_mats_list, pair_names_master = [], None
            for reps_m, anchors_m in zip(reps_multi_all, anchors_multi_all):
                mats_dict, pair_names = layer_layer_similarity_on_anchors(
                    reps_multi=reps_m, reps_single=reps_s,
                    anchors_multi=anchors_m, anchors_single=anchors_s
                )
                if pair_names_master is None:
                    pair_names_master = pair_names
                pair_mats_list.append(mats_dict)

            avg_mats = {}
            for (km, ks) in (pair_names_master or []):
                key = f"{km}__{ks}"
                stack = np.stack([d[key] for d in pair_mats_list if key in d], axis=0)
                avg_mats[key] = stack.mean(axis=0)

            key_bucket = f"single_{x}-of-{y}"
            combined.setdefault(key_bucket, {})
            for pair_key, A in avg_mats.items():
                combined[key_bucket].setdefault(pair_key, []).append(A)

    # ================= 汇总输出 =================
    for key_bucket, pair_bucket in combined.items():
        for pair_key, mats in pair_bucket.items():
            arr = np.stack(mats, axis=0)
            A_global = arr.mean(axis=0)
            Lm, Ls = A_global.shape
            tag = f"{key_bucket}__{pair_key}_Lm{Lm}_Ls{Ls}"
            np.save(combined_dir / f"{tag}.npy", A_global)
            plot_layer_layer_matrix(
                A=A_global,
                save_path=combined_dir / f"{tag}.png",
                title=f"[Combined] {key_bucket} [{pair_key}]",
                cmap=args.cmap
            )
            print(f"[Combined saved] {combined_dir / (tag + '.npy')}")


if __name__ == "__main__":
    main()
