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
MODULE_NAMES = ["base", "mlp", "attn"]


def is_llama_family_name(model_name: str) -> bool:
    lowered = model_name.lower()
    return "llama" in lowered or "meta-llama" in lowered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch last-hop single-hop activations into multi-hop prompts."
    )
    parser.add_argument("--input_file", required=True, help="Path to input JSON case file.")
    parser.add_argument("--output_dir", default="results/{model_name}/last_hop_repair")
    parser.add_argument("--model_name", default=DEFAULT_GPTJ_REMOTE)
    parser.add_argument("--local", action="store_true")
    parser.add_argument(
        "--patch_position",
        choices=["last_k", "subject_last"],
        default="last_k",
        help="Whether to patch the last-k prompt-end tokens or the last subject token.",
    )
    parser.add_argument("--last_k", type=int, default=3)
    parser.add_argument(
        "--min_token_offset",
        type=int,
        default=1,
        help="For patch_position=last_k, only run token offsets in [min_token_offset, last_k].",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to run, e.g. 5,10,15. Defaults to 5,10,15,...",
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


def parse_layer_list(layer_text: Optional[str], num_layers: int) -> List[int]:
    if layer_text is not None:
        values = []
        for piece in layer_text.split(","):
            piece = piece.strip()
            if not piece:
                continue
            layer = int(piece)
            if 0 <= layer < num_layers:
                values.append(layer)
        values = sorted(set(values))
        if not values:
            raise ValueError("No valid layers found in --layers")
        return values

    values = list(range(5, num_layers, 5))
    if not values:
        values = [min(num_layers - 1, 0)] if num_layers > 0 else []
    return values


def check_contains(pred: str, gold_list: List[str]) -> bool:
    lowered = pred.lower()
    return any(g and g.lower() in lowered for g in gold_list)


def untuple(x):
    return x[0] if isinstance(x, tuple) else x


def get_pad_token_id(tokenizer):
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


def module_kind(module_name: str) -> Optional[str]:
    return None if module_name == "base" else module_name


def encode_text(mt: ModelAndTokenizer, text: str) -> List[int]:
    if mt.is_llama_family:
        return mt.tokenizer.encode(text)
    return mt.tokenizer.encode(text)


def encode_without_special_tokens(mt: ModelAndTokenizer, text: str) -> List[int]:
    if mt.is_llama_family:
        return mt.tokenizer.encode(text, add_special_tokens=False)
    return mt.tokenizer.encode(text, add_special_tokens=False)


def tokenize_prompt(mt: ModelAndTokenizer, prompt: str) -> Dict[str, torch.Tensor]:
    if mt.is_llama_family:
        return mt.tokenizer(prompt, return_tensors="pt").to(mt.device)

    token_ids = encode_text(mt, prompt)
    attention_mask = [1] * len(token_ids)
    return {
        "input_ids": torch.tensor([token_ids], device=mt.device),
        "attention_mask": torch.tensor([attention_mask], device=mt.device),
    }


def decode_tokens(mt: ModelAndTokenizer, token_array) -> List[str]:
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(mt, row) for row in token_array]
    return [mt.tokenizer.decode([int(t)], skip_special_tokens=False) for t in token_array]


def get_prepend_space(mt: ModelAndTokenizer) -> bool:
    name = getattr(mt.tokenizer, "name_or_path", "")
    return "llama-3" in name.lower() or "meta-llama-3" in name.lower()


def find_token_span_by_search(
    mt: ModelAndTokenizer,
    input_ids: torch.Tensor,
    substring: str,
    prepend_space: bool = False,
) -> Optional[Tuple[int, int]]:
    candidate_strings = [substring]
    if prepend_space and not substring.startswith(" "):
        candidate_strings.append(" " + substring)

    input_ids = torch.as_tensor(input_ids)
    for candidate in candidate_strings:
        substring_tokens = mt.tokenizer(
            candidate,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]
        substring_tokens = substring_tokens.to(input_ids.device)
        length = len(substring_tokens)
        if length == 0:
            continue
        for start in range(len(input_ids) - length + 1):
            end = start + length
            if torch.all(input_ids[start:end] == substring_tokens):
                return (start, end)
    return None


def find_token_span_by_offsets(
    mt: ModelAndTokenizer, text: str, substring: str
) -> Optional[Tuple[int, int]]:
    try:
        enc = mt.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except Exception:
        return None
    offsets = enc.get("offset_mapping")
    if offsets is None:
        return None
    char_loc = text.find(substring)
    if char_loc < 0:
        return None
    start_char = char_loc
    end_char = char_loc + len(substring)
    tok_start, tok_end = None, None
    for i, (s, e) in enumerate(offsets):
        if e <= s:
            continue
        if tok_start is None and s <= start_char < e:
            tok_start = i
        if tok_start is not None and s < end_char <= e:
            tok_end = i + 1
            break
        if tok_start is not None and s >= start_char and e >= end_char:
            tok_end = i + 1
            break
    if tok_start is None:
        for i, (s, e) in enumerate(offsets):
            if e <= s:
                continue
            if s < end_char and e > start_char:
                tok_start = i
                break
    if tok_start is not None and tok_end is None:
        for i in range(tok_start, len(offsets)):
            s, e = offsets[i]
            if e >= end_char:
                tok_end = i + 1
                break
    if tok_start is None or tok_end is None:
        return None
    return (tok_start, tok_end)


def find_token_range(mt: ModelAndTokenizer, text_or_tokens, substring: str) -> Optional[Tuple[int, int]]:
    if mt.is_llama_family:
        text = str(text_or_tokens)
        encoded = mt.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded.input_ids[0]
        prepend_space = get_prepend_space(mt)

        match = find_token_span_by_search(mt, input_ids, substring, prepend_space=False)
        if match is None and prepend_space:
            match = find_token_span_by_search(mt, input_ids, substring, prepend_space=True)
        if match is None:
            match = find_token_span_by_offsets(mt, text, substring)
        return match

    toks = decode_tokens(mt, text_or_tokens)
    whole_string = "".join(toks)
    char_loc = whole_string.find(substring)
    if char_loc < 0:
        return None
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    if tok_start is None or tok_end is None:
        return None
    return (tok_start, tok_end)


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


def get_prompt_token_count(mt: ModelAndTokenizer, prompt: str) -> int:
    inputs = tokenize_prompt(mt, prompt)
    return int(inputs["input_ids"].shape[1])


def resolve_donor_subject(case: dict) -> Optional[str]:
    orig = case.get("orig") or {}
    triples_labeled = orig.get("triples_labeled") or []
    if triples_labeled:
        last_triple = triples_labeled[-1]
        if isinstance(last_triple, (list, tuple)) and len(last_triple) > 0:
            donor_subject = last_triple[0]
            if donor_subject:
                return str(donor_subject)

    single_hops = case.get("single_hops") or []
    if len(single_hops) >= 2:
        donor_subject = single_hops[-2].get("answer")
        if donor_subject:
            return str(donor_subject)

    return None


def collect_donor_states(
    mt: ModelAndTokenizer,
    prompt: str,
    patch_position: str,
    last_k: int,
    subject: Optional[str] = None,
) -> Dict[str, object]:
    inputs = tokenize_prompt(mt, prompt)
    token_count = int(inputs["input_ids"].shape[1])
    effective_k = min(last_k, token_count)
    subject_last_index = None
    if patch_position == "subject_last":
        if subject is None:
            raise ValueError("subject must be provided for subject_last mode")
        subject_range = find_token_range(mt, prompt if mt.is_llama_family else inputs["input_ids"][0], subject)
        if subject_range is None:
            raise ValueError(f"Could not locate subject in donor prompt: {subject!r}")
        subject_last_index = subject_range[1] - 1

    trace_layers = []
    for module_name in MODULE_NAMES:
        kind = module_kind(module_name)
        for layer in range(mt.num_layers):
            trace_layers.append(layername(mt.model, layer, kind))
    cache = {module_name: {} for module_name in MODULE_NAMES}
    with torch.no_grad(), nethook.TraceDict(mt.model, trace_layers, clone=True, detach=True) as td:
        mt.model(**inputs, use_cache=True, return_dict=True)
    for module_name in MODULE_NAMES:
        kind = module_kind(module_name)
        for layer in range(mt.num_layers):
            lname = layername(mt.model, layer, kind)
            output = untuple(td[lname].output)[0]
            if patch_position == "subject_last":
                cache[module_name][layer] = output[subject_last_index : subject_last_index + 1].detach().cpu()
            else:
                cache[module_name][layer] = output[-effective_k:].detach().cpu()
    return {
        "states": cache,
        "effective_k": effective_k,
        "prompt_token_count": token_count,
        "subject_last_index": subject_last_index,
    }


def generate_from_patched_state(
    mt: ModelAndTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values,
    logits: torch.Tensor,
    max_new_tokens: int,
) -> str:
    eos_token_id = mt.tokenizer.eos_token_id
    generated_tokens = []
    current_logits = logits[:, -1, :]
    current_past = past_key_values
    current_attention_mask = attention_mask

    for _ in range(max_new_tokens):
        next_token = torch.argmax(current_logits, dim=-1, keepdim=True)
        token_id = int(next_token[0, 0].item())
        if eos_token_id is not None and token_id == eos_token_id:
            break
        generated_tokens.append(token_id)
        current_attention_mask = torch.cat(
            [
                current_attention_mask,
                torch.ones(
                    (current_attention_mask.shape[0], 1),
                    dtype=current_attention_mask.dtype,
                    device=current_attention_mask.device,
                ),
            ],
            dim=1,
        )
        with torch.no_grad():
            outputs = mt.model(
                input_ids=next_token,
                attention_mask=current_attention_mask,
                past_key_values=current_past,
                use_cache=True,
                return_dict=True,
            )
        current_past = outputs.past_key_values
        current_logits = outputs.logits[:, -1, :]

    if not generated_tokens:
        return ""
    return mt.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def patched_prefill_forward(
    mt: ModelAndTokenizer,
    prompt: str,
    subject: str,
    donor_cache: dict,
    patch_position: str,
    token_offset: Optional[int],
    layer: int,
    module_name: str,
):
    kind = module_kind(module_name)
    inputs = tokenize_prompt(mt, prompt)
    prompt_token_count = int(inputs["input_ids"].shape[1])

    if patch_position == "subject_last":
        subject_range = find_token_range(mt, prompt if mt.is_llama_family else inputs["input_ids"][0], subject)
        if subject_range is None:
            raise ValueError(f"Could not locate subject in target prompt: {subject!r}")
        target_subject_last_index = subject_range[1] - 1
        donor_tensor = donor_cache["states"][module_name][layer][0]
        patch_index = target_subject_last_index
    else:
        if token_offset is None:
            raise ValueError("token_offset must be provided when patch_position=last_k")
        donor_tensor = donor_cache["states"][module_name][layer][-(token_offset)]
        patch_index = prompt_token_count - token_offset
    target_layer = layername(mt.model, layer, kind)

    def edit_output(output, layer):
        if layer != target_layer:
            return output
        hidden = untuple(output)
        if hidden.shape[1] != prompt_token_count:
            return output
        if patch_index < 0 or hidden.shape[1] <= patch_index:
            return output
        hidden[:, patch_index, :] = donor_tensor.to(hidden.device, dtype=hidden.dtype)
        return output

    with torch.no_grad(), nethook.TraceDict(mt.model, [target_layer], edit_output=edit_output):
        outputs = mt.model(
            **inputs,
            use_cache=True,
            return_dict=True,
        )
    return inputs, outputs


def patch_generation(
    mt: ModelAndTokenizer,
    prompt: str,
    subject: str,
    donor_cache: dict,
    patch_position: str,
    token_offset: Optional[int],
    layer: int,
    module_name: str,
    max_new_tokens: int,
) -> str:
    inputs, outputs = patched_prefill_forward(
        mt,
        prompt,
        subject,
        donor_cache,
        patch_position=patch_position,
        token_offset=token_offset,
        layer=layer,
        module_name=module_name,
    )
    return generate_from_patched_state(
        mt,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        past_key_values=outputs.past_key_values,
        logits=outputs.logits,
        max_new_tokens=max_new_tokens,
    )


def evaluate_case(
    mt: ModelAndTokenizer,
    case: dict,
    patch_position: str,
    last_k: int,
    min_token_offset: int,
    max_new_tokens: int,
    layers_to_run: List[int],
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
        "subject": case.get("subject"),
        "source_question_index": case.get("source_question_index"),
        "single_hop_question": donor["question"],
        "multi_hop_question": target_question,
        "donor_generation": donor_generation,
        "donor_is_correct": donor_correct,
        "baseline_generation": baseline_generation,
        "baseline_is_correct": baseline_correct,
        "patch_position_mode": patch_position,
        "last_k_requested": last_k,
        "layers_to_run": layers_to_run,
    }
    if not donor_correct:
        print(
            f"[Skip] donor single-hop incorrect | case_id={case.get('case_id')} "
            f"known_id={case.get('known_id')} | question={donor['question']}"
        )
        return meta, []
    if baseline_correct:
        print(
            f"[Skip] baseline multi-hop already correct | case_id={case.get('case_id')} "
            f"known_id={case.get('known_id')} | question={target_question}"
        )
        return meta, []

    donor_subject = None
    if patch_position == "subject_last":
        donor_subject = resolve_donor_subject(case)
        if not donor_subject:
            print(
                f"[Skip] invalid subject_last case: missing donor subject | "
                f"case_id={case.get('case_id')} known_id={case.get('known_id')}"
            )
            return meta, []

    try:
        donor_cache = collect_donor_states(
            mt,
            donor_prompt,
            patch_position=patch_position,
            last_k=last_k,
            subject=donor_subject,
        )
    except ValueError as e:
        print(
            f"[Skip] subject not found in donor prompt | case_id={case.get('case_id')} "
            f"known_id={case.get('known_id')} | {e}"
        )
        return meta, []
    effective_k = min(donor_cache["effective_k"], get_prompt_token_count(mt, target_prompt))
    rows = []
    if patch_position == "subject_last":
        patch_specs = [(None, "subject_last", case["subject"])]
    else:
        start_offset = max(1, min_token_offset)
        if start_offset > effective_k:
            meta["effective_k"] = effective_k
            meta["min_token_offset"] = start_offset
            return meta, []
        patch_specs = [
            (token_offset, f"last_{token_offset}", case["subject"])
            for token_offset in range(start_offset, effective_k + 1)
        ]

    for token_offset, patch_label, subject in patch_specs:
        for module_name in MODULE_NAMES:
            for layer in layers_to_run:
                try:
                    patched_generation = patch_generation(
                        mt,
                        target_prompt,
                        subject,
                        donor_cache,
                        patch_position=patch_position,
                        token_offset=token_offset,
                        layer=layer,
                        module_name=module_name,
                        max_new_tokens=max_new_tokens,
                    )
                except ValueError as e:
                    print(
                        f"[Skip row] subject not found in target prompt | case_id={case.get('case_id')} "
                        f"known_id={case.get('known_id')} | {e}"
                    )
                    continue
                patched_correct = check_contains(patched_generation, target_golds)
                rows.append(
                    {
                        **meta,
                        "token_offset": 0 if token_offset is None else -token_offset,
                        "patch_label": patch_label,
                        "module": module_name,
                        "layer": layer,
                        "patched_generation": patched_generation,
                        "patched_is_correct": patched_correct,
                        "is_repaired": (not baseline_correct) and patched_correct,
                    }
                )
    meta["effective_k"] = effective_k
    meta["min_token_offset"] = max(1, min_token_offset)
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


def save_case_result(cases_dir: Path, meta: dict, rows: List[dict]) -> None:
    cases_dir.mkdir(parents=True, exist_ok=True)
    filename = build_case_filename(meta)
    payload = {"meta": meta, "rows": rows}
    with (cases_dir / filename).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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
    layers_to_run = parse_layer_list(args.layers, mt.num_layers)
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
        "layers_to_run": layers_to_run,
    }

    for case in tqdm(sliced, desc="Running last-hop repair"):
        meta, rows = evaluate_case(
            mt,
            case,
            patch_position=args.patch_position,
            last_k=args.last_k,
            min_token_offset=args.min_token_offset,
            max_new_tokens=args.max_new_tokens,
            layers_to_run=layers_to_run,
        )
        save_case_result(cases_dir, meta, rows)
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