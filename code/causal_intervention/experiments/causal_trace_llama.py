import argparse
import json
import os
import re
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import KnownsDataset
from rome.tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)
from util import nethook
from util.globals import DATA_DIR
from util.runningstats import Covariance, tally


PROMPT_PATCH_TOKEN_COUNT = 10
DEFAULT_LLAMA3_PATH = "/data/xkliu/hf_models/Meta-Llama-3-8B-Instruct"
DEFAULT_LLAMA3_REMOTE = "meta-llama/Meta-Llama-3-8B-Instruct"


def main():
    parser = argparse.ArgumentParser(description="Causal Tracing for Meta-Llama-3-8B-Instruct")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    def parse_noise_rule(code):
        if code in ["m", "s"]:
            return code
        elif re.match("^[uts][\d\.]+", code):
            return code
        else:
            return float(code)

    aa(
        "--model_name",
        default=DEFAULT_LLAMA3_REMOTE,
        help="Local path or HF name for the Llama 3 model.",
    )
    aa("--local", action="store_true", help="Load model from local path instead of remote HF name.")
    aa("--fact_file", default=None)
    aa("--output_dir", default="results/{model_name}/causal_trace")
    aa("--noise_level", default="s3", type=parse_noise_rule)
    aa("--replace", default=0, type=int)
    args = parser.parse_args()

    modeldir = f'r{args.replace}_{args.model_name.replace("/", "_")}'
    modeldir = f"n{args.noise_level}_" + modeldir
    output_dir = args.output_dir.format(model_name=modeldir)
    result_dir = f"{output_dir}/cases"
    pdf_dir = f"{output_dir}/pdfs"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    torch_dtype = torch.float16
    mt = ModelAndTokenizer(args.model_name, torch_dtype=torch_dtype, local=args.local)

    if args.fact_file is None:
        knowns = KnownsDataset(DATA_DIR)
    else:
        with open(args.fact_file) as f:
            knowns = json.load(f)

    noise_level = args.noise_level
    uniform_noise = False
    if isinstance(noise_level, str):
        if noise_level.startswith("s"):
            factor = float(noise_level[1:]) if len(noise_level) > 1 else 1.0
            noise_level = factor * collect_embedding_std(
                mt, [k["subject"] for k in knowns]
            )
            print(f"Using noise_level {noise_level} to match model times {factor}")
        elif noise_level == "m":
            noise_level = collect_embedding_gaussian(mt)
            print("Using multivariate gaussian to match model noise")
        elif noise_level.startswith("t"):
            degrees = float(noise_level[1:])
            noise_level = collect_embedding_tdist(mt, degrees)
        elif noise_level.startswith("u"):
            uniform_noise = True
            noise_level = float(noise_level[1:])

    for knowledge in tqdm(knowns):
        known_id = knowledge["known_id"]
        for kind in None, "mlp", "attn":
            kind_suffix = f"_{kind}" if kind else ""
            filename = f"{result_dir}/knowledge_{known_id}{kind_suffix}.npz"
            if not os.path.isfile(filename):
                result = calculate_hidden_flow(
                    mt,
                    knowledge["prompt"],
                    knowledge["subject"],
                    expect=knowledge["attribute"],
                    answer_aliases=knowledge.get("answer_alias", []),
                    kind=kind,
                    noise=noise_level,
                    uniform_noise=uniform_noise,
                    replace=args.replace,
                )
                numpy_result = {
                    k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                    for k, v in result.items()
                }
                numpy.savez(filename, **numpy_result)
            else:
                numpy_result = numpy.load(filename, allow_pickle=True)
            if not numpy_result["correct_prediction"]:
                tqdm.write(f"Skipping {knowledge['prompt']}")
                continue
            plot_result = dict(numpy_result)
            plot_result["kind"] = kind
            pdfname = f'{pdf_dir}/{str(numpy_result["answer"]).strip()}_{known_id}{kind_suffix}.pdf'
            if known_id > 200:
                continue
            plot_trace_heatmap(plot_result, savepdf=pdfname, modelname="Llama")


def trace_with_patch(
    model,
    inp,
    states_to_patch,
    answers_t,
    tokens_to_mix,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    trace_layers=None,
    answer_position=None,
):
    rs = numpy.random.RandomState(1)
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def corrupt_rep(x, layer):
        if layer == embed_layername and tokens_to_mix is not None:
            b, e = tokens_to_mix
            noise_data = noise_fn(
                torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
            ).to(x.device, dtype=x.dtype)
            if replace:
                x[1:, b:e] = noise_data
            else:
                x[1:, b:e] += noise_data
        return x

    additional_layers = [] if trace_layers is None else trace_layers
    donor_layers = [embed_layername] + list(patch_spec.keys())
    first_pass_trace = None
    if patch_spec:
        with torch.no_grad(), nethook.TraceDict(
            model,
            donor_layers,
            edit_output=corrupt_rep,
        ) as td:
            model(**inp)
            first_pass_trace = td

    def inject_rep(x, layer):
        if layer == embed_layername or layer not in patch_spec:
            return x
        h = untuple(x)
        donor_h = untuple(first_pass_trace[layer].output)
        for t in patch_spec[layer]:
            h[1:, t] = donor_h[1:, t]
        return x

    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=inject_rep,
    ) as td:
        outputs_exp = model(**inp)

    target_position = -1 if answer_position is None else answer_position - 1
    probs = torch.softmax(outputs_exp.logits[1:, target_position, :], dim=1).mean(dim=0)[
        answers_t
    ]

    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs


def trace_with_repatch(
    model,
    inp,
    states_to_patch,
    states_to_unpatch,
    answers_t,
    tokens_to_mix,
    noise=0.1,
    uniform_noise=False,
):
    rs = numpy.random.RandomState(1)
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    unpatch_spec = defaultdict(list)
    for t, l in states_to_unpatch:
        unpatch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    def patch_rep(x, layer):
        if layer == embed_layername:
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device, dtype=x.dtype)
            return x
        if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
            return x
        h = untuple(x)
        for t in patch_spec.get(layer, []):
            h[1:, t] = h[0, t]
        for t in unpatch_spec.get(layer, []):
            h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
        return x

    for first_pass in [True, False] if states_to_unpatch else [False]:
        with torch.no_grad(), nethook.TraceDict(
            model,
            [embed_layername] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
            edit_output=patch_rep,
        ) as td:
            outputs_exp = model(**inp)
            if first_pass:
                first_pass_trace = td

    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]
    return probs


def calculate_hidden_flow(
    mt,
    prompt,
    subject,
    samples=10,
    noise=0.1,
    token_range=None,
    uniform_noise=False,
    replace=False,
    window=10,
    kind=None,
    expect=None,
    answer_aliases=None,
):
    answer_aliases = answer_aliases or []
    formatted_prompt = format_qa_prompt(prompt)
    generated_text = generate_answer(mt, formatted_prompt)
    golds = [expect] + list(answer_aliases) if expect is not None else list(answer_aliases)
    match = find_answer_match(generated_text, golds)
    if match is None:
        return dict(correct_prediction=False, generated_text=generated_text)

    answer, answer_char_range = match
    prefix_text = formatted_prompt + generated_text[: answer_char_range[0]]
    inp = make_inputs(mt.tokenizer, [prefix_text] * (samples + 1), device=mt.device)
    answer_token_ids = encode_without_special_tokens(mt.tokenizer, answer)
    if not answer_token_ids:
        return dict(correct_prediction=False, generated_text=generated_text)

    answer_t = answer_token_ids[0]
    answer_position = inp["input_ids"].shape[1]
    with torch.no_grad():
        base_score = answer_prob_at_position(mt.model, inp, answer_t, answer_position)[0].item()

    prompt_token_ids = encode_text(mt.tokenizer, formatted_prompt)
    prompt_token_len = len(prompt_token_ids)
    prompt_subject_range = find_token_range(mt.tokenizer, formatted_prompt, subject)
    prompt_last_token_index = prompt_token_len - 1

    e_range = find_token_range(mt.tokenizer, prefix_text, subject)
    if prompt_subject_range is None or e_range is None:
        return dict(correct_prediction=False, generated_text=generated_text)

    if token_range == "subject_last":
        token_range = [e_range[1] - 1]
    else:
        if token_range is None:
            token_range = range(prompt_subject_range[0], prompt_last_token_index + 1)
        else:
            raise ValueError(f"Unknown token_range: {token_range}")

    low_score = trace_with_patch(
        mt.model,
        inp,
        [],
        answer_t,
        e_range,
        noise=noise,
        uniform_noise=uniform_noise,
        answer_position=answer_position,
    ).item()
    if not kind:
        differences = trace_important_states(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            answer_position=answer_position,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            token_range=token_range,
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            answer_position=answer_position,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            window=window,
            kind=kind,
            token_range=token_range,
        )
    differences = differences.detach().cpu()
    differences = (base_score - differences).detach().cpu()
    prompt_tokens = decode_tokens(mt.tokenizer, prompt_token_ids)
    traced_labels = prompt_tokens[prompt_subject_range[0] : prompt_last_token_index + 1]
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        num_subject_tokens=prompt_subject_range[1] - prompt_subject_range[0],
        traced_labels=traced_labels,
        answer=answer,
        window=window,
        correct_prediction=True,
        kind=kind or "",
    )


def trace_important_states(
    model,
    num_layers,
    inp,
    e_range,
    answer_t,
    answer_position=None,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    ntoks = inp["input_ids"].shape[1]
    table = []

    if token_range is None:
        token_range = range(ntoks)
    for tnum in token_range:
        row = []
        for layer in range(num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
                answer_position=answer_position,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    model,
    num_layers,
    inp,
    e_range,
    answer_t,
    kind,
    answer_position=None,
    window=10,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    ntoks = inp["input_ids"].shape[1]
    table = []

    if token_range is None:
        token_range = range(ntoks)
    for tnum in token_range:
        row = []
        for layer in range(num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model,
                inp,
                layerlist,
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
                answer_position=answer_position,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


class ModelAndTokenizer:
    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
        local=False,
    ):
        local_model_paths = {
            "Meta-Llama-3-8B-Instruct": DEFAULT_LLAMA3_PATH,
            DEFAULT_LLAMA3_REMOTE: DEFAULT_LLAMA3_PATH,
        }
        model_path = local_model_paths.get(model_name, model_name) if local else model_name

        if tokenizer is None:
            assert model_path is not None
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        if model is None:
            assert model_path is not None
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=low_cpu_mem_usage,
                torch_dtype=torch_dtype,
            )
            nethook.set_requires_grad(False, model)
            model.eval().cuda()
        self.tokenizer = tokenizer
        self.model = model
        self.device = next(model.parameters()).device
        self.layer_names = [
            n
            for n, _ in model.named_modules()
            if re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", n)
            or re.match(r"^model\.layers\.\d+$", n)
        ]
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def layername(model, num, kind=None):
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


def guess_subject(prompt):
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()


def plot_hidden_flow(
    mt,
    prompt,
    subject=None,
    samples=10,
    noise=0.1,
    uniform_noise=False,
    window=10,
    kind=None,
    savepdf=None,
):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt,
        prompt,
        subject,
        samples=samples,
        noise=noise,
        uniform_noise=uniform_noise,
        window=window,
        kind=kind,
    )
    plot_trace_heatmap(result, savepdf, modelname="Llama")


def plot_trace_heatmap(result, savepdf=None, title=None, xlabel=None, modelname=None):
    differences = result["scores"]
    answer = result["answer"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result.get("traced_labels", result["input_tokens"]))

    fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
    h = ax.pcolor(
        differences,
        cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[kind],
        vmin=0,
    )
    ax.invert_yaxis()
    ax.set_yticks([0.5 + i for i in range(len(differences))])
    ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
    ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
    ax.set_yticklabels(labels)
    if not modelname:
        modelname = "Llama"
    if not kind:
        ax.set_title("Impact of injecting corrupted states into clean run")
        ax.set_xlabel(f"single restored layer within {modelname}")
    else:
        kindname = "MLP" if kind == "mlp" else "Attn"
        ax.set_title(f"Impact of injecting corrupted {kindname} states into clean run")
        ax.set_xlabel(f"center of interval of {window} patched {kindname} layers")
    cb = plt.colorbar(h)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    elif answer is not None:
        cb.ax.set_title(f"Δp({str(answer).strip()})", y=-0.16, fontsize=10)
    if savepdf:
        os.makedirs(os.path.dirname(savepdf), exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_all_flow(mt, prompt, subject=None):
    for kind in ["mlp", "attn", None]:
        plot_hidden_flow(mt, prompt, subject, kind=kind)


def format_qa_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def check_contains(pred: str, gold_list: List[str]) -> bool:
    p = pred.lower()
    return any(g and g.lower() in p for g in gold_list)


def generate_answer(mt, prompt: str, max_new_tokens: int = 128) -> str:
    inputs = mt.tokenizer(prompt, return_tensors="pt").to(mt.model.device)
    pad_token_id = mt.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = mt.tokenizer.eos_token_id
    with torch.no_grad():
        outputs = mt.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return mt.tokenizer.decode(generated_ids, skip_special_tokens=True)


def find_answer_match(text: str, candidates: List[str]) -> Optional[Tuple[str, Tuple[int, int]]]:
    lowered_text = text.lower()
    matches = []
    for candidate in candidates:
        if not candidate:
            continue
        lowered_candidate = candidate.lower()
        start = lowered_text.find(lowered_candidate)
        if start != -1:
            matches.append((start, start + len(candidate), candidate))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], -len(item[2])))
    start, end, candidate = matches[0]
    return candidate, (start, end)


def answer_prob_at_position(model, inp, answer_t, answer_position):
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, answer_position - 1, :], dim=1)
    return probs[:, answer_t]


def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [encode_text(tokenizer, p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    pad_id = get_pad_token_id(tokenizer)
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([int(t)], skip_special_tokens=False) for t in token_array]


def encode_text(tokenizer, text):
    return tokenizer.encode(text)


def encode_without_special_tokens(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def get_pad_token_id(tokenizer):
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    if "[PAD]" in tokenizer.all_special_tokens:
        return tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    return 0


def get_prepend_space(tokenizer):
    name = getattr(tokenizer, "name_or_path", "")
    return "llama-3" in name.lower() or "meta-llama-3" in name.lower()


def find_token_span_by_search(tokenizer, input_ids, substring, prepend_space=False):
    candidate_strings = [substring]
    if prepend_space and not substring.startswith(" "):
        candidate_strings.append(" " + substring)

    input_ids = torch.as_tensor(input_ids)
    for candidate in candidate_strings:
        substring_tokens = tokenizer(
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


def find_token_span_by_offsets(tokenizer, text, substring):
    try:
        enc = tokenizer(
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


def find_token_range(tokenizer, text, substring):
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded.input_ids[0]
    prepend_space = get_prepend_space(tokenizer)

    match = find_token_span_by_search(tokenizer, input_ids, substring, prepend_space=False)
    if match is None and prepend_space:
        match = find_token_span_by_search(tokenizer, input_ids, substring, prepend_space=True)
    if match is None:
        match = find_token_span_by_offsets(tokenizer, text, substring)
    return match


def predict_token(mt, prompts, return_p=False):
    inp = make_inputs(mt.tokenizer, prompts, device=mt.device)
    preds, p = predict_from_input(mt.model, inp)
    result = [mt.tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    return result


def predict_from_input(model, inp):
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p


def collect_embedding_std(mt, subjects):
    alldata = []
    for s in subjects:
        inp = make_inputs(mt.tokenizer, [s], device=mt.device)
        with nethook.Trace(mt.model, layername(mt.model, 0, "embed")) as t:
            mt.model(**inp)
            alldata.append(t.output[0])
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()
    return noise_level


def get_embedding_cov(mt):
    model = mt.model
    tokenizer = mt.tokenizer

    def get_ds():
        ds_name = "wikitext"
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name],
        )
        maxlen = getattr(model.config, "max_position_embeddings", None)
        if maxlen is None:
            maxlen = getattr(model.config, "n_positions", 100)
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    ds = get_ds()
    sample_size = 1000
    batch_size = 5
    filename = None
    batch_tokens = 100

    stat = Covariance()
    loader = tally(
        stat,
        ds,
        cache=filename,
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=0,
    )
    with torch.no_grad():
        for batch_group in loader:
            for batch in batch_group:
                batch = dict_to_(batch, "cuda")
                if "position_ids" in batch:
                    del batch["position_ids"]
                with nethook.Trace(model, layername(mt.model, 0, "embed")) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                stat.add(feats.cpu().double())
    return stat.mean(), stat.covariance()


def make_generator_transform(mean=None, cov=None):
    d = len(mean) if mean is not None else len(cov)
    device = mean.device if mean is not None else cov.device
    layer = torch.nn.Linear(d, d, dtype=torch.double)
    nethook.set_requires_grad(False, layer)
    layer.to(device)
    layer.bias[...] = 0 if mean is None else mean
    if cov is None:
        layer.weight[...] = torch.eye(d).to(device)
    else:
        _, s, v = cov.svd()
        w = s.sqrt()[None, :] * v
        layer.weight[...] = w
    return layer


def collect_embedding_gaussian(mt):
    m, c = get_embedding_cov(mt)
    return make_generator_transform(m, c)


def collect_embedding_tdist(mt, degree=3):
    u_sample = torch.from_numpy(
        numpy.random.RandomState(2).chisquare(df=degree, size=1000)
    )
    fixed_sample = ((degree - 2) / u_sample).sqrt()
    mvg = collect_embedding_gaussian(mt)

    def normal_to_student(x):
        gauss = mvg(x)
        size = gauss.shape[:-1].numel()
        factor = fixed_sample[:size].reshape(gauss.shape[:-1] + (1,))
        student = factor * gauss
        return student

    return normal_to_student


if __name__ == "__main__":
    main()