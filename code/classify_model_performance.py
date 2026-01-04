# -*- coding: utf-8 -*-
"""
classify_model_performance_split.py

改进版：
- 按 multi-hop 问题正确性拆分 case
- 每个 case 拆成多个子 case（multi-hop 全正确部分 / 全错误部分）
- 保留 4 类分类逻辑
"""

import json
from pathlib import Path
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from collections import Counter
from copy import deepcopy
import random
import numpy as np

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def generate_answer(model, tokenizer, question: str, device: str, max_new_tokens: int = 512, template=True) -> str:
    if template:
        question = f"Question: {question}\nAnswer:"
    inputs = tokenizer(question, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False
        )
    ans = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return ans


def check_contains(pred: str, gold_list: List[str]) -> bool:
    p = pred.lower()
    return any(g.lower() in p for g in gold_list)


def evaluate_case(case: dict, model, tokenizer, device: str) -> dict:
    """评估单个 case 的多跳与单跳"""
    result = {
        "case_id": case.get("case_id"),
        "multi_details": [],
        "single_details": [],
        "single_correct": True,
    }

    # multi-hop
    for q in case["questions"]:
        pred = generate_answer(model, tokenizer, q, device)
        # print(f"Q: {q}\nA: {pred}\n---")
        golds = [case["answer"]] + case.get("answer_alias", [])
        ok = check_contains(pred, golds)
        result["multi_details"].append({"question": q, "pred": pred, "ok": ok})

    # single-hop
    for sh in case["single_hops"]:
        q = sh["question"]
        pred = generate_answer(model, tokenizer, q, device)
        golds = [sh["answer"]] + sh.get("answer_alias", [])
        ok = check_contains(pred, golds)
        result["single_details"].append({"question": q, "pred": pred, "ok": ok})
        if not ok:
            result["single_correct"] = False

    return result


def assign_to_class(new_case, single_correct: bool, multi_correct: bool, classified: dict):
    """根据 single/multi 正确性放入四类"""
    if single_correct and multi_correct:
        cls = 1
    elif (not single_correct) and multi_correct:
        cls = 2
    elif single_correct and (not multi_correct):
        cls = 3
    else:
        cls = 4
    classified[cls].append(new_case)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="EleutherAI/gpt-j-6B")
    parser.add_argument("--output_dir", type=str, default="./outputs_split")
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--mode", type=str, default="strict",
                    choices=["strict", "separate"],
                    help="Data sample: strict for Llama; separate for GPT-J;")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for torch / cuda / python"
    )

    args = parser.parse_args()
    set_seeds(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]

    classified = {1: [], 2: [], 3: [], 4: []}

    for i, case in enumerate(data):
        if args.max_cases and i >= args.max_cases:
            break
        print(f"\n[{i+1}/{len(data)}] Evaluating case {case.get('case_id')}")
        res = evaluate_case(case, model, tokenizer, device)

        multi_correct_list = [m for m in res["multi_details"] if m["ok"]]
        multi_wrong_list = [m for m in res["multi_details"] if not m["ok"]]
        single_correct = res["single_correct"]

        # --- 多跳正确部分 ---
        if multi_correct_list:
            new_case = deepcopy(case)
            new_case["questions"] = [m["question"] for m in multi_correct_list]
            assign_to_class(new_case, single_correct, True, classified)

        # strict 模式（当前默认行为）
        if args.mode == "strict":
            # 如果单跳全对且有多跳正确 ⇒ 跳过错误部分
            if single_correct and multi_correct_list:
                continue

        # --- 多跳错误部分 ---
        if multi_wrong_list:
            new_case = deepcopy(case)
            new_case["questions"] = [m["question"] for m in multi_wrong_list]
            assign_to_class(new_case, single_correct, False, classified)


    # --- 保存 ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = {
        1: "class_1_both_correct",
        2: "class_2_single_wrong_multi_correct",
        3: "class_3_single_correct_multi_wrong",
        4: "class_4_both_wrong"
    }

    for i in range(1, 5):
        Path(out_dir / f"{names[i]}.json").write_text(
            json.dumps(classified[i], indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    # --- 统计 ---
    counts = {i: len(classified[i]) for i in range(1, 5)}
    hop_distrib = {i: Counter(len(c["single_hops"]) for c in classified[i]) for i in range(1, 5)}

    print("\n=== 分类统计（拆分后） ===")
    for i in range(1, 5):
        print(f"{names[i]}: {counts[i]} 个子样本")
        print(f"  跳数分布: {dict(hop_distrib[i])}")

    Path(out_dir / "summary.json").write_text(
        json.dumps({"counts": counts, "hop_distribution": hop_distrib}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\n结果已保存至 {out_dir}")


if __name__ == "__main__":
    main()
