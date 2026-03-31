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
MODULE_KINDS = [None, "mlp", "attn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch last-hop single-hop activations into multi-hop prompts."
    )
    parser.add_argument("--input_file", required=True, help="Path to input JSON case file.")
    parser.add_argument("--output_dir", default="results/{model_name}/last_hop_repair")
    parser.add_argument("--model_name", default=DEFAULT_GPTJ_REMOTE)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--last_k", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
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
        local_model_paths = {
            DEFAULT_GPTJ_REMOTE: DEFAULT_GPTJ_LOCAL,
            "gpt-j-6B": DEFAULT_GPTJ_LOCAL,
            DEFAULT_LLAMA3_REMOTE: DEFAULT_LLAMA3_LOCAL,
            "Meta-Llama-3-8B-Instruct": DEFAULT_LLAMA3_LOCAL,
        }
        model_path = local_model_paths.get(model_name, model_name) if local else model_name
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
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
        self.layer_names = [
            n
            for n, _ in model.named_modules()
            if re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", n)
            or re.match(r"^model\.layers\.\d+$", n)
        ]
        self.num_layers = len(self.layer_names)


def layername(model, num: int, kind: Optional[str] = None) -> str:
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        if kind == "embed":
            return "model.embed_tokens"
        if kind == "attn":
            kind = "self_attn"
        return f'model.layers.{num}{"" if kind is None else "." + kind}'
    raise AssertionError("unknown transformer structure")


def format_qa_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def load_cases(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def check_contains(pred: str, gold_list: List[str]) -> bool:
    lowered = pred.lower()
    return any(g and g.lower() in lowered for g in gold_list)


def untuple(x):
    return x[0] if isinstance(x, tuple) else x


def get_pad_token_id(tokenizer):
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


def generate_answer(mt: ModelAndTokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = mt.tokenizer(prompt, return_tensors="pt").to(mt.device)
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


def get_prompt_token_count(mt: ModelAndTokenizer, prompt: str) -> int:
    return len(mt.tokenizer.encode(prompt))


def module_label(kind: Optional[str]) -> str:
    return "base" if kind is None else kind


def collect_donor_states(mt: ModelAndTokenizer, prompt: str, last_k: int) -> Dict[str, object]:
    token_count = get_prompt_token_count(mt, prompt)
    effective_k = min(last_k, token_count)
    trace_layers = []
    for kind in MODULE_KINDS:
        for layer in range(mt.num_layers):
            trace_layers.append(layername(mt.model, layer, kind))
    inputs = mt.tokenizer(prompt, return_tensors="pt").to(mt.device)
    cache = {module_label(kind): {} for kind in MODULE_KINDS}
    with torch.no_grad(), nethook.TraceDict(mt.model, trace_layers, clone=True, detach=True) as td:
        mt.model(**inputs)
    for kind in MODULE_KINDS:
        label = module_label(kind)
        for layer in range(mt.num_layers):
            lname = layername(mt.model, layer, kind)
            output = untuple(td[lname].output)[0]
            cache[label][layer] = output[-effective_k:].detach().cpu()
    return {
        "states": cache,
        "effective_k": effective_k,
        "prompt_token_count": token_count,
    }


def patch_generation(
    mt: ModelAndTokenizer,
    prompt: str,
    donor_cache: dict,
    token_offset: int,
    layer: int,
    kind: Optional[str],
    max_new_tokens: int,
) -> str:
    module = module_label(kind)
    donor_tensor = donor_cache["states"][module][layer][-(token_offset)]
    prompt_token_count = get_prompt_token_count(mt, prompt)
    patch_index = prompt_token_count - token_offset
    target_layer = layername(mt.model, layer, kind)

    def edit_output(output, layer):
        if layer != target_layer:
            return output
        hidden = untuple(output)
        if patch_index < 0 or hidden.shape[1] <= patch_index:
            return output
        hidden[:, patch_index, :] = donor_tensor.to(hidden.device, dtype=hidden.dtype)
        return output

    inputs = mt.tokenizer(prompt, return_tensors="pt").to(mt.device)
    with torch.no_grad(), nethook.TraceDict(mt.model, [target_layer], edit_output=edit_output):
        outputs = mt.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=get_pad_token_id(mt.tokenizer),
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return mt.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def evaluate_case(
    mt: ModelAndTokenizer,
    case: dict,
    last_k: int,
    max_new_tokens: int,
) -> Tuple[dict, List[dict]]:
    donor = case["single_hops"][-1]
    donor_prompt = format_qa_prompt(donor["question"])
    donor_golds = [donor["answer"]] + donor.get("answer_alias", [])
    donor_generation = generate_answer(mt, donor_prompt, max_new_tokens=max_new_tokens)
    donor_correct = check_contains(donor_generation, donor_golds)

    target_question = case.get("prompt") or case["questions"][case.get("source_question_index", 0)]
    target_prompt = format_qa_prompt(target_question)
    target_golds = [case.get("attribute", case.get("answer", ""))] + case.get("answer_alias", [])
    baseline_generation = generate_answer(mt, target_prompt, max_new_tokens=max_new_tokens)
    baseline_correct = check_contains(baseline_generation, target_golds)

    meta = {
        "case_id": case.get("case_id"),
        "known_id": case.get("known_id"),
        "source_question_index": case.get("source_question_index"),
        "single_hop_question": donor["question"],
        "multi_hop_question": target_question,
        "donor_generation": donor_generation,
        "donor_is_correct": donor_correct,
        "baseline_generation": baseline_generation,
        "baseline_is_correct": baseline_correct,
        "last_k_requested": last_k,
    }
    if (not donor_correct) or baseline_correct:
        return meta, []

    donor_cache = collect_donor_states(mt, donor_prompt, last_k=last_k)
    effective_k = min(donor_cache["effective_k"], get_prompt_token_count(mt, target_prompt))
    rows = []
    for token_offset in range(1, effective_k + 1):
        for kind in MODULE_KINDS:
            for layer in range(mt.num_layers):
                patched_generation = patch_generation(
                    mt,
                    target_prompt,
                    donor_cache,
                    token_offset=token_offset,
                    layer=layer,
                    kind=kind,
                    max_new_tokens=max_new_tokens,
                )
                patched_correct = check_contains(patched_generation, target_golds)
                rows.append(
                    {
                        **meta,
                        "token_offset": -token_offset,
                        "module": module_label(kind),
                        "layer": layer,
                        "patched_generation": patched_generation,
                        "patched_is_correct": patched_correct,
                        "is_repaired": patched_correct,
                    }
                )
    meta["effective_k"] = effective_k
    return meta, rows


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    modeldir = args.model_name.replace("/", "_")
    output_dir = Path(args.output_dir.format(model_name=modeldir))
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

    all_rows = []
    summaries = []
    counts = {
        "total_cases": len(sliced),
        "donor_correct": 0,
        "baseline_wrong": 0,
        "eligible_cases": 0,
        "repaired_cases": 0,
        "repair_rows": 0,
    }

    for case in tqdm(sliced, desc="Running last-hop repair"):
        meta, rows = evaluate_case(mt, case, last_k=args.last_k, max_new_tokens=args.max_new_tokens)
        summaries.append(meta)
        if meta["donor_is_correct"]:
            counts["donor_correct"] += 1
        if not meta["baseline_is_correct"]:
            counts["baseline_wrong"] += 1
        if rows:
            counts["eligible_cases"] += 1
            if any(row["is_repaired"] for row in rows):
                counts["repaired_cases"] += 1
            counts["repair_rows"] += len(rows)
            all_rows.extend(rows)

    save_jsonl(output_dir / "results.jsonl", all_rows)
    save_json(output_dir / "case_summary.json", summaries)
    save_json(output_dir / "summary.json", counts)
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()