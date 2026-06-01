import ast
import sys
from pathlib import Path

import pandas as pd
import torch


PROJECT_CODE_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE_DIR))

from model_support import (
    attention_layer_names,
    attention_modules,
    get_model_family,
    greedy_generation_kwargs,
    layer_names,
    load_causal_lm,
    load_tokenizer as support_load_tokenizer,
    mlp_layer_names,
    norm_module,
    should_prepend_space_for_token_search,
)


HF_TOKEN = ""


def load_tokenizer(model_name):
    return support_load_tokenizer(model_name, token=HF_TOKEN or None)


def load_model(model_name, device="cuda"):
    if "70b" in model_name or "70B" in model_name:
        return load_causal_lm(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            token=HF_TOKEN or None,
        )
    return load_causal_lm(model_name, device=device, token=HF_TOKEN or None)


def get_layer_names(model):
    return layer_names(model)


def get_attention_layers_names(model):
    return attention_layer_names(model)


def get_mlp_layers_names(model):
    return mlp_layer_names(model)


def get_attention_modules(model, layer, k=0):
    return attention_modules(model, layer, k)


def get_norm_module(model):
    return norm_module(model)


def get_prepend_space(model):
    return should_prepend_space_for_token_search(model)


def decode_generated(tokenizer, generated, prompts):
    text = tokenizer.batch_decode(generated, skip_special_tokens=True)
    pred = [t[len(p):].strip().replace("\n", " ") for t, p in zip(text, prompts)]
    return pred


def get_answers(entry, key):
    aliases = ast.literal_eval(entry[f"{key}_aliases"])
    aliases = [a for a in aliases if len(a) > 1]
    entity = entry[f"{key}_label"]
    return {
        f"{key}_answers": [entity] + aliases,
    }


def check_answer_in_pred(pred, answers):
    pred = pred.lower()
    return any([a.lower() in pred for a in answers])


def generate_and_test_answers(model, tokenizer, prompts, answers):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=10,
            **greedy_generation_kwargs(tokenizer),
        )
    predictions = decode_generated(tokenizer, generated, prompts)
    for prompt, pred, a in zip(prompts, predictions, answers):
        print(f"Prompt: {prompt}\nPrediction: {pred}\nAnswers: {a}\nResults: {check_answer_in_pred(pred, a)}")
        print("-----------------------------------")
    return [check_answer_in_pred(p, a) for p, a in zip(predictions, answers)]


def print_dataset_statistics(dataset):
    print("Dataset statistics:")
    for target in ["e1", "r1", "e2", "r2", "e3"]:
        proportions = dataset[f"{target}_type"].value_counts(normalize=True) * 100
        counts = dataset[f"{target}_type"].value_counts()
        stats = pd.merge(counts, proportions, left_index=True, right_index=True)
        print(stats)


def rebalance_dataset(df, key="e2_type", size=100, secondary_key=None):
    if secondary_key is None:
        balanced = df.groupby(key).apply(lambda x: x.sample(min(size, len(x)))).reset_index(drop=True)
    else:
        balanced = df.groupby([key, secondary_key]).apply(lambda x: x.sample(min(size // 2, len(x)))).reset_index(drop=True)
    balanced = balanced.sort_values("id")
    return balanced


def last_relation_word(relation):
    return relation.split(" ")[-3]
