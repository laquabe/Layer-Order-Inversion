import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_CODE_DIR = CURRENT_DIR.parent.parent
if str(PROJECT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE_DIR))


from latent_patch.util import nethook


DEFAULT_GPTJ_REMOTE = "EleutherAI/gpt-j-6B"
DEFAULT_GPTJ_LOCAL = "/data/xkliu/hf_models/gpt-j-6b"
DEFAULT_LLAMA3_REMOTE = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_LLAMA3_LOCAL = "/data/xkliu/hf_models/Meta-Llama-3-8B-Instruct"


def is_llama_family_name(model_name: str) -> bool:
    lowered = model_name.lower()
    return "llama" in lowered or "meta-llama" in lowered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run context expansion experiments on multi-hop QA prompts."
    )
    parser.add_argument("--input_file", required=True, help="Path to input JSON case file.")
    parser.add_argument("--output_dir", default="results/{model_name}/context_expansion")
    parser.add_argument("--model_name", default=DEFAULT_GPTJ_REMOTE)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--context_mode",
        choices=["filler", "entity"],
        default="filler",
        help="Use repeated filler text or entity/triple-derived context.",
    )
    parser.add_argument(
        "--filler_text",
        default="Let me think.",
        help="Base filler unit used in filler mode.",
    )
    parser.add_argument(
        "--filler_repeat",
        type=int,
        default=None,
        help="Manual filler repeat count. If unset, auto-align filler length to entity context length.",
    )
    parser.add_argument(
        "--check_single_hops",
        action="store_true",
        help="Require all single-hop questions to be answered correctly before expansion.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class ModelAndTokenizer:
    def __init__(
        self,
        model_name: str,
        local: bool = False,
        torch_dtype=None,
        low_cpu_mem_usage: bool = False,
    ):
        self.model_name = model_name
        self.is_llama_family = is_llama_family_name(model_name)
        local_model_paths = {
            DEFAULT_GPTJ_REMOTE: DEFAULT_GPTJ_LOCAL,
            "gpt-j-6B": DEFAULT_GPTJ_LOCAL,
            DEFAULT_LLAMA3_REMOTE: DEFAULT_LLAMA3_LOCAL,
            "Meta-Llama-3-8B-Instruct": DEFAULT_LLAMA3_LOCAL,
        }
        model_path = local_model_paths.get(model_name, model_name) if local else model_name

        if self.is_llama_family:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        nethook.set_requires_grad(False, model)
        model.eval().cuda()
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device


def format_qa_prompt(question: str, context: Optional[str] = None) -> str:
    question = question.strip()
    if context:
        context = context.strip()
        return f"Question: {question} {context}\nAnswer:"
    return f"Question: {question}\nAnswer:"


def load_cases(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def slugify_subject(text: Optional[str]) -> str:
    if not text:
        return "unknown_subject"
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text or "unknown_subject"


def build_case_filename(meta: dict) -> str:
    known_id = meta.get("known_id")
    known_part = f"knowledge_{known_id}" if known_id is not None else "knowledge_unknown"
    subject_part = slugify_subject(meta.get("subject"))
    return f"{known_part}_{subject_part}.json"


def check_contains(pred: str, gold_list: List[str]) -> bool:
    lowered = pred.lower()
    return any(g and g.lower() in lowered for g in gold_list)


def encode_text(mt: ModelAndTokenizer, text: str) -> List[int]:
    return mt.tokenizer.encode(text)


def tokenize_prompt(mt: ModelAndTokenizer, prompt: str) -> Dict[str, torch.Tensor]:
    if mt.is_llama_family:
        return mt.tokenizer(prompt, return_tensors="pt").to(mt.device)

    token_ids = encode_text(mt, prompt)
    attention_mask = [1] * len(token_ids)
    return {
        "input_ids": torch.tensor([token_ids], device=mt.device),
        "attention_mask": torch.tensor([attention_mask], device=mt.device),
    }


def get_pad_token_id(tokenizer):
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


def generate_answer(mt: ModelAndTokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenize_prompt(mt, prompt)
    with torch.no_grad():
        outputs = mt.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=get_pad_token_id(mt.tokenizer),
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return mt.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def token_length(mt: ModelAndTokenizer, text: str) -> int:
    return len(encode_text(mt, text))


def get_target_question(case: dict) -> str:
    return case.get("prompt") or case["questions"][case.get("source_question_index", 0)]


def get_multi_hop_golds(case: dict) -> List[str]:
    return [case.get("attribute", case.get("answer", ""))] + case.get("answer_alias", [])


def triple_to_sentence(triple: List[str]) -> str:
    if len(triple) < 3:
        return " ".join(str(x) for x in triple).strip()
    subject, relation, obj = triple[:3]
    return f"{subject} {relation} {obj}.".strip()


def build_entity_context(case: dict) -> str:
    triples = case.get("orig", {}).get("triples_labeled", [])
    sentences = [triple_to_sentence(triple) for triple in triples if triple]
    return " ".join(sentence for sentence in sentences if sentence).strip()


def build_repeated_filler(text: str, repeat: int) -> str:
    repeat = max(0, repeat)
    if repeat == 0:
        return ""
    return " ".join([text.strip()] * repeat).strip()


def determine_auto_filler_repeat(mt: ModelAndTokenizer, filler_text: str, entity_context: str) -> int:
    if not filler_text.strip() or not entity_context.strip():
        return 0

    target_len = token_length(mt, entity_context)
    if target_len <= 0:
        return 0

    repeat = 0
    while True:
        next_repeat = repeat + 1
        next_text = build_repeated_filler(filler_text, next_repeat)
        next_len = token_length(mt, next_text)
        if next_len > target_len:
            break
        repeat = next_repeat
    return repeat


def build_contexts(
    mt: ModelAndTokenizer,
    case: dict,
    filler_text: str,
    filler_repeat: Optional[int],
) -> Dict[str, object]:
    entity_context = build_entity_context(case)
    entity_tokens = token_length(mt, entity_context) if entity_context else 0
    if filler_repeat is None:
        auto_repeat = determine_auto_filler_repeat(mt, filler_text, entity_context)
        filler_repeat_used = auto_repeat
        filler_repeat_source = "auto_aligned"
    else:
        filler_repeat_used = max(0, filler_repeat)
        filler_repeat_source = "manual"
    filler_context = build_repeated_filler(filler_text, filler_repeat_used)
    filler_tokens = token_length(mt, filler_context) if filler_context else 0

    return {
        "entity_context": entity_context,
        "entity_context_token_count": entity_tokens,
        "filler_context": filler_context,
        "filler_context_token_count": filler_tokens,
        "filler_repeat_used": filler_repeat_used,
        "filler_repeat_source": filler_repeat_source,
    }


def evaluate_single_hops(mt: ModelAndTokenizer, case: dict, max_new_tokens: int) -> Tuple[bool, List[dict]]:
    details = []
    all_correct = True
    for hop in case.get("single_hops", []):
        prompt = format_qa_prompt(hop["question"])
        generation = generate_answer(mt, prompt, max_new_tokens=max_new_tokens)
        golds = [hop["answer"]] + hop.get("answer_alias", [])
        correct = check_contains(generation, golds)
        details.append(
            {
                "question": hop["question"],
                "prompt": prompt,
                "generation": generation,
                "gold_answers": golds,
                "is_correct": correct,
            }
        )
        if not correct:
            all_correct = False
    return all_correct, details


def evaluate_case(
    mt: ModelAndTokenizer,
    case: dict,
    context_mode: str,
    filler_text: str,
    filler_repeat: Optional[int],
    max_new_tokens: int,
    check_single_hops: bool,
) -> dict:
    target_question = get_target_question(case)
    target_golds = get_multi_hop_golds(case)
    baseline_prompt = format_qa_prompt(target_question)
    baseline_generation = generate_answer(mt, baseline_prompt, max_new_tokens=max_new_tokens)
    baseline_correct = check_contains(baseline_generation, target_golds)

    context_info = build_contexts(mt, case, filler_text=filler_text, filler_repeat=filler_repeat)
    selected_context = context_info[f"{context_mode}_context"]
    expanded_prompt = format_qa_prompt(target_question, selected_context)
    expanded_generation = None
    expanded_correct = None

    single_hop_passed = None
    single_hop_details: List[dict] = []
    if check_single_hops:
        single_hop_passed, single_hop_details = evaluate_single_hops(
            mt, case, max_new_tokens=max_new_tokens
        )

    eligible = (not baseline_correct) and ((single_hop_passed is not False) if check_single_hops else True)
    if eligible:
        expanded_generation = generate_answer(mt, expanded_prompt, max_new_tokens=max_new_tokens)
        expanded_correct = check_contains(expanded_generation, target_golds)

    return {
        "case_id": case.get("case_id"),
        "known_id": case.get("known_id"),
        "subject": case.get("subject"),
        "source_question_index": case.get("source_question_index"),
        "multi_hop_question": target_question,
        "gold_answers": target_golds,
        "context_mode": context_mode,
        "baseline_prompt": baseline_prompt,
        "baseline_generation": baseline_generation,
        "baseline_is_correct": baseline_correct,
        "entity_context": context_info["entity_context"],
        "entity_context_token_count": context_info["entity_context_token_count"],
        "filler_text": filler_text,
        "filler_context": context_info["filler_context"],
        "filler_context_token_count": context_info["filler_context_token_count"],
        "filler_repeat_used": context_info["filler_repeat_used"],
        "filler_repeat_source": context_info["filler_repeat_source"],
        "selected_context": selected_context,
        "selected_context_token_count": context_info[f"{context_mode}_context_token_count"],
        "expanded_prompt": expanded_prompt,
        "expanded_generation": expanded_generation,
        "expanded_is_correct": expanded_correct,
        "check_single_hops": check_single_hops,
        "single_hops_passed": single_hop_passed,
        "single_hop_details": single_hop_details,
        "eligible": eligible,
        "is_repaired": eligible and bool(expanded_correct),
    }


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_case_result(cases_dir: Path, result: dict) -> None:
    cases_dir.mkdir(parents=True, exist_ok=True)
    filename = build_case_filename(result)
    with (cases_dir / filename).open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    modeldir = args.model_name.replace("/", "_")
    output_dir = Path(args.output_dir.format(model_name=modeldir))
    cases_dir = output_dir / "cases"
    results_file = output_dir / "results.jsonl"
    if results_file.exists() and not args.overwrite:
        raise FileExistsError(f"{results_file} already exists. Use --overwrite to replace it.")
    output_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = ("llama" in args.model_name.lower()) or ("gpt-j" in args.model_name.lower())
    mt = ModelAndTokenizer(
        args.model_name,
        local=args.local,
        torch_dtype=torch.float16 if use_fp16 else None,
    )

    cases = load_cases(args.input_file)
    sliced = cases[args.start_index :]
    if args.max_cases is not None:
        sliced = sliced[: args.max_cases]

    results = []
    counts = {
        "total_cases": len(sliced),
        "context_mode": args.context_mode,
        "check_single_hops": args.check_single_hops,
        "multi_hop_baseline_wrong": 0,
        "passed_single_hop_check": 0,
        "eligible_cases": 0,
        "expanded_correct_cases": 0,
        "expanded_accuracy": None,
        "repaired_cases": 0,
        "skipped_baseline_correct": 0,
        "skipped_single_hop_check_failed": 0,
    }

    for case in tqdm(sliced, desc="Running context expansion"):
        result = evaluate_case(
            mt,
            case,
            context_mode=args.context_mode,
            filler_text=args.filler_text,
            filler_repeat=args.filler_repeat,
            max_new_tokens=args.max_new_tokens,
            check_single_hops=args.check_single_hops,
        )
        save_case_result(cases_dir, result)
        results.append(result)

        if not result["baseline_is_correct"]:
            counts["multi_hop_baseline_wrong"] += 1
        else:
            counts["skipped_baseline_correct"] += 1

        if args.check_single_hops and result["single_hops_passed"]:
            counts["passed_single_hop_check"] += 1
        if args.check_single_hops and result["single_hops_passed"] is False:
            counts["skipped_single_hop_check_failed"] += 1

        if result["eligible"]:
            counts["eligible_cases"] += 1
            if result["expanded_is_correct"]:
                counts["expanded_correct_cases"] += 1
                counts["repaired_cases"] += 1

    if counts["eligible_cases"] > 0:
        counts["expanded_accuracy"] = counts["expanded_correct_cases"] / counts["eligible_cases"]

    save_jsonl(results_file, results)
    save_json(output_dir / "case_summary.json", results)
    save_json(output_dir / "summary.json", counts)
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()