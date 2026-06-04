# -*- coding: utf-8 -*-
"""
classify_model_performance_split.py

改进版：
- 按 multi-hop 问题正确性拆分 case
- 每个 case 拆成多个子 case（multi-hop 全正确部分 / 全错误部分）
- 保留 4 类分类逻辑
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import torch
from collections import Counter
from copy import deepcopy
import random
import numpy as np

from model_support import (
    default_torch_dtype,
    greedy_generation_kwargs,
    load_causal_lm,
    load_tokenizer,
)

CLASS_NAMES = {
    1: "class_1_both_correct",
    2: "class_2_single_wrong_multi_correct",
    3: "class_3_single_correct_multi_wrong",
    4: "class_4_both_wrong",
}


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
            **greedy_generation_kwargs(tokenizer),
        )
    ans = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return ans


def check_contains(pred: str, gold_list: List[str]) -> bool:
    p = pred.lower()
    return any(g.lower() in p for g in gold_list)


def evaluate_case(case: dict, model, tokenizer, device: str, max_new_tokens: int) -> dict:
    """评估单个 case 的多跳与单跳"""
    result = {
        "case_id": case.get("case_id"),
        "multi_details": [],
        "single_details": [],
        "single_correct": True,
    }

    # multi-hop
    for q in case["questions"]:
        pred = generate_answer(model, tokenizer, q, device, max_new_tokens=max_new_tokens)
        # print(f"Q: {q}\nA: {pred}\n---")
        golds = [case["answer"]] + case.get("answer_alias", [])
        ok = check_contains(pred, golds)
        result["multi_details"].append({"question": q, "pred": pred, "ok": ok})

    # single-hop
    for sh in case["single_hops"]:
        q = sh["question"]
        pred = generate_answer(model, tokenizer, q, device, max_new_tokens=max_new_tokens)
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


def case_key(input_index: int, case_id) -> Tuple[int, str]:
    return input_index, str(case_id)


def load_checkpoint_rows(checkpoint_path: Path) -> Dict[Tuple[int, str], dict]:
    rows: Dict[Tuple[int, str], dict] = {}
    if not checkpoint_path.exists():
        return rows

    with checkpoint_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping invalid checkpoint line {line_no}: {exc}")
                continue
            key = case_key(row["input_index"], row.get("case_id"))
            rows[key] = row
    return rows


def append_checkpoint_row(checkpoint_path: Path, row: dict) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def build_checkpoint_row(input_index: int, case: dict, result: dict, args) -> dict:
    return {
        "input_index": input_index,
        "case_id": case.get("case_id"),
        "case": case,
        "evaluation": result,
        "run": {
            "model": args.model,
            "local": args.local,
            "max_new_tokens": args.max_new_tokens,
            "mode": args.mode,
            "seed": args.seed,
            "max_cases": args.max_cases,
        },
    }


def add_result_to_classified(row: dict, data: List[dict], mode: str, classified: dict) -> None:
    input_index = row["input_index"]
    case = row.get("case") or data[input_index]
    res = row["evaluation"]

    multi_correct_list = [m for m in res["multi_details"] if m["ok"]]
    multi_wrong_list = [m for m in res["multi_details"] if not m["ok"]]
    single_correct = res["single_correct"]

    if multi_correct_list:
        new_case = deepcopy(case)
        new_case["questions"] = [m["question"] for m in multi_correct_list]
        assign_to_class(new_case, single_correct, True, classified)

    if mode == "strict" and single_correct and multi_correct_list:
        return

    if multi_wrong_list:
        new_case = deepcopy(case)
        new_case["questions"] = [m["question"] for m in multi_wrong_list]
        assign_to_class(new_case, single_correct, False, classified)


def extract_subject(case: dict):
    return case["orig"]["triples_labeled"][0][0]


def hop_count(case: dict) -> int:
    return len(case["orig"]["triples_labeled"])


def expand_case_records_to_questions(case_records: List[dict]) -> Dict[int, List[dict]]:
    grouped = {2: [], 3: [], 4: []}
    known_id = 0

    for case in case_records:
        hop = hop_count(case)
        if hop not in grouped:
            continue

        subject = extract_subject(case)
        answer = case.get("answer", "")
        for question_index, question in enumerate(case.get("questions", [])):
            record = deepcopy(case)
            record["known_id"] = known_id
            record["subject"] = subject
            record["prompt"] = question
            record["attribute"] = answer
            record["source_question_index"] = question_index
            grouped[hop].append(record)
            known_id += 1

    return grouped


def selected_case_keys(data: List[dict], max_cases: Optional[int]) -> Set[Tuple[int, str]]:
    selected = data if max_cases is None else data[:max_cases]
    return {case_key(i, case.get("case_id")) for i, case in enumerate(selected)}


def write_case_level_outputs(case_level_dir: Path, classified: dict) -> dict:
    case_level_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, 5):
        Path(case_level_dir / f"{CLASS_NAMES[i]}.json").write_text(
            json.dumps(classified[i], indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    counts = {i: len(classified[i]) for i in range(1, 5)}
    hop_distrib = {i: Counter(len(c["single_hops"]) for c in classified[i]) for i in range(1, 5)}
    summary = {
        "counts": counts,
        "hop_distribution": hop_distrib,
    }
    Path(case_level_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    return summary


def write_question_level_outputs(ques_level_dir: Path, classified: dict) -> dict:
    ques_level_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for cls_id in range(1, 5):
        class_name = CLASS_NAMES[cls_id]
        grouped = expand_case_records_to_questions(classified[cls_id])
        summary[class_name] = {}

        for hop in (2, 3, 4):
            records = grouped[hop]
            output_path = ques_level_dir / f"{class_name}_{hop}hop.json"
            output_path.write_text(
                json.dumps(records, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            summary[class_name][f"{hop}hop"] = len(records)

    Path(ques_level_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    return summary


def write_final_outputs(out_dir: Path, rows_by_key: Dict[Tuple[int, str], dict], data: List[dict], args) -> None:
    wanted_keys = selected_case_keys(data, args.max_cases)
    rows = [
        row
        for key, row in rows_by_key.items()
        if key in wanted_keys
    ]
    rows.sort(key=lambda row: row["input_index"])

    classified = {1: [], 2: [], 3: [], 4: []}
    for row in rows:
        add_result_to_classified(row, data, args.mode, classified)

    case_summary = write_case_level_outputs(out_dir / "case_level", classified)
    ques_summary = write_question_level_outputs(out_dir / "ques_level", classified)

    print("\n=== 分类统计（拆分后） ===")
    for i in range(1, 5):
        print(f"{CLASS_NAMES[i]}: {case_summary['counts'][i]} 个 case-level 子样本")
        print(f"  case-level 跳数分布: {dict(case_summary['hop_distribution'][i])}")
        print(f"  ques-level 数量: {ques_summary[CLASS_NAMES[i]]}")

    print(f"\ncase-level 结果已保存至 {out_dir / 'case_level'}")
    print(f"ques-level 结果已保存至 {out_dir / 'ques_level'}")


def remove_previous_outputs(out_dir: Path, checkpoint_path: Path) -> None:
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    for dirname in ("case_level", "ques_level"):
        path = out_dir / dirname
        if path.exists():
            shutil.rmtree(path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="EleutherAI/gpt-j-6B")
    parser.add_argument("--local", action="store_true", help="Resolve --model through local model aliases.")
    parser.add_argument("--output_dir", type=str, default="./outputs_split")
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--mode", type=str, default="strict",
                    choices=["strict", "separate"],
                    help="strict skips wrong multi-hop variants when a case also has correct multi-hop variants; separate keeps both and yields more mixed-case samples.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing checkpoint/final outputs and start from scratch.")
    parser.add_argument("--rebuild_only", "--rebuild-only", action="store_true", help="Only rebuild class JSON files from case_results.jsonl without loading a model.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for torch / cuda / python"
    )

    args = parser.parse_args()
    set_seeds(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "case_results.jsonl"

    if args.overwrite and args.rebuild_only:
        raise ValueError("--overwrite cannot be used with --rebuild_only")
    if args.overwrite:
        remove_previous_outputs(out_dir, checkpoint_path)

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]

    rows_by_key = load_checkpoint_rows(checkpoint_path)
    if args.rebuild_only:
        if not rows_by_key:
            raise FileNotFoundError(f"No checkpoint rows found at {checkpoint_path}")
        write_final_outputs(out_dir, rows_by_key, data, args)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.model, local=args.local)
    model = load_causal_lm(
        args.model,
        local=args.local,
        torch_dtype=default_torch_dtype(args.model, device),
        device=device,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    target_total = len(data) if args.max_cases is None else min(args.max_cases, len(data))
    print(f"Loaded {len(rows_by_key)} checkpointed case(s) from {checkpoint_path}")

    for i, case in enumerate(data):
        if args.max_cases is not None and i >= args.max_cases:
            break
        key = case_key(i, case.get("case_id"))
        if key in rows_by_key:
            print(f"\n[{i+1}/{target_total}] Skipping checkpointed case {case.get('case_id')}")
            continue

        print(f"\n[{i+1}/{target_total}] Evaluating case {case.get('case_id')}")
        res = evaluate_case(case, model, tokenizer, device, max_new_tokens=args.max_new_tokens)
        row = build_checkpoint_row(i, case, res, args)
        append_checkpoint_row(checkpoint_path, row)
        rows_by_key[key] = row
        print(f"[Checkpoint] Saved case {case.get('case_id')} to {checkpoint_path}")

    write_final_outputs(out_dir, rows_by_key, data, args)


if __name__ == "__main__":
    main()
