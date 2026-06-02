from __future__ import annotations

import re
from typing import Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_GPTJ_REMOTE = "EleutherAI/gpt-j-6B"
DEFAULT_GPTJ_LOCAL = "/data/xkliu/hf_models/gpt-j-6b"
DEFAULT_LLAMA3_REMOTE = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_LLAMA3_LOCAL = "/data/xkliu/hf_models/Meta-Llama-3-8B-Instruct"
DEFAULT_QWEN3_REMOTE = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_QWEN3_LOCAL = "/data/xkliu/hf_models/Qwen3-4B-Instruct-2507"
DEFAULT_QWEN3_14B_REMOTE = "Qwen/Qwen3-14B"
DEFAULT_QWEN3_14B_LOCAL = "/data/xkliu/hf_models/Qwen3-14B"

LOCAL_MODEL_PATHS = {
    DEFAULT_GPTJ_REMOTE: DEFAULT_GPTJ_LOCAL,
    "gpt-j-6B": DEFAULT_GPTJ_LOCAL,
    DEFAULT_LLAMA3_REMOTE: DEFAULT_LLAMA3_LOCAL,
    "Meta-Llama-3-8B-Instruct": DEFAULT_LLAMA3_LOCAL,
    DEFAULT_QWEN3_REMOTE: DEFAULT_QWEN3_LOCAL,
    "Qwen3-4B-Instruct-2507": DEFAULT_QWEN3_LOCAL,
    DEFAULT_QWEN3_14B_REMOTE: DEFAULT_QWEN3_14B_LOCAL,
    "Qwen3-14B": DEFAULT_QWEN3_14B_LOCAL,
    "qwen3-14b": DEFAULT_QWEN3_14B_LOCAL,
}

MODEL_LAYERS_FAMILIES = {"llama", "gemma", "qwen3"}


def resolve_model_path(model_name: str, local: bool = False) -> str:
    if not local:
        return model_name
    return LOCAL_MODEL_PATHS.get(model_name, model_name)


def get_model_family(model_or_name: Any) -> str:
    config = getattr(model_or_name, "config", None)
    model_type = getattr(config, "model_type", None)
    if model_type:
        lowered_type = model_type.lower()
        if lowered_type in {"qwen3", "qwen3_moe"}:
            return "qwen3"
        if "llama" in lowered_type:
            return "llama"
        if "gemma" in lowered_type:
            return "gemma"
        if lowered_type in {"gptj", "gpt_j"}:
            return "gptj"
        if lowered_type == "gpt2":
            return "gpt2"
        if "gpt_neox" in lowered_type:
            return "gpt_neox"

    name = str(model_or_name)
    lowered = name.lower()
    if "qwen3" in lowered:
        return "qwen3"
    if "llama" in lowered:
        return "llama"
    if "gemma" in lowered:
        return "gemma"
    if "gpt-j" in lowered or "gptj" in lowered:
        return "gptj"
    if "gpt-neox" in lowered or "gpt_neox" in lowered:
        return "gpt_neox"
    if "gpt2" in lowered:
        return "gpt2"
    return "unknown"


def uses_model_layers(model_or_name: Any) -> bool:
    return get_model_family(model_or_name) in MODEL_LAYERS_FAMILIES


def should_use_fast_tokenizer(model_or_name: Any) -> bool:
    return get_model_family(model_or_name) in MODEL_LAYERS_FAMILIES


def should_left_pad(model_or_name: Any) -> bool:
    return get_model_family(model_or_name) in MODEL_LAYERS_FAMILIES


def should_prepend_space_for_token_search(model_or_name: Any) -> bool:
    family = get_model_family(model_or_name)
    if family in {"gptj", "gpt2", "gpt_neox"}:
        return True
    if family == "llama":
        name = str(getattr(getattr(model_or_name, "config", None), "_name_or_path", model_or_name))
        if name.lower() == "llama":
            return True
        return "llama-3" in name.lower() or "meta-llama-3" in name.lower()
    if family == "gemma":
        return True
    if family == "qwen3":
        return False
    return False


def configure_tokenizer(tokenizer, model_or_name: Any):
    family = get_model_family(model_or_name)
    if family == "qwen3":
        if tokenizer.pad_token is None:
            pad_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
            if pad_id is not None and pad_id != tokenizer.unk_token_id:
                tokenizer.pad_token = "<|endoftext|>"
        tokenizer.padding_side = "left"
        return tokenizer

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if should_left_pad(model_or_name):
        tokenizer.padding_side = "left"
    return tokenizer


def load_tokenizer(model_name: str, local: bool = False, token: Optional[str] = None):
    model_path = resolve_model_path(model_name, local=local)
    kwargs = {}
    if token:
        kwargs["token"] = token
    if should_use_fast_tokenizer(model_name):
        kwargs["use_fast"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    return configure_tokenizer(tokenizer, model_name)


def load_causal_lm(
    model_name: str,
    local: bool = False,
    torch_dtype=None,
    low_cpu_mem_usage: bool = False,
    device: Optional[str] = None,
    device_map: Optional[str] = None,
    token: Optional[str] = None,
):
    model_path = resolve_model_path(model_name, local=local)
    kwargs = {
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if device_map is not None:
        kwargs["device_map"] = device_map
    if token:
        kwargs["token"] = token

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if device is not None and device_map is None:
        model = model.to(device)
    return model


def default_torch_dtype(model_or_name: Any, device: str):
    if device != "cuda":
        return torch.float32
    family = get_model_family(model_or_name)
    if family in {"gptj", "llama", "gemma", "qwen3"}:
        return torch.float16
    return torch.float32


def get_pad_token_id(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")


def get_eos_token_id(tokenizer):
    return tokenizer.eos_token_id


def greedy_generation_kwargs(tokenizer) -> dict:
    # 不显式传 eos_token_id：与原版各实验一致，交由 model.generation_config 决定停止条件
    # （Llama-3 的 eos 是列表 [128001,128009]，Qwen3 是 [151645,151643]，显式只传单个 id 会改变停止行为）。
    return {
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": get_pad_token_id(tokenizer),
    }


def encode_text(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text)


def encode_without_special_tokens(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def decode_tokens(tokenizer, token_array) -> list[str]:
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([int(t)], skip_special_tokens=False) for t in token_array]


def layer_names(model) -> list[str]:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return [f"transformer.h.{i}" for i in range(model.config.num_hidden_layers)]
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return [f"gpt_neox.layers.{i}" for i in range(model.config.num_hidden_layers)]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return [f"model.layers.{i}" for i in range(model.config.num_hidden_layers)]
    raise ValueError(f"Model type {type(model)} not supported")


def attention_layer_names(model) -> list[str]:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return [f"transformer.h.{i}.attn" for i in range(model.config.num_hidden_layers)]
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return [f"gpt_neox.layers.{i}.attention" for i in range(model.config.num_hidden_layers)]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return [f"model.layers.{i}.self_attn" for i in range(model.config.num_hidden_layers)]
    raise ValueError(f"Model type {type(model)} not supported")


def mlp_layer_names(model) -> list[str]:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return [f"transformer.h.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return [f"gpt_neox.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    raise ValueError(f"Model type {type(model)} not supported")


def attention_modules(model, layer: int, k: int = 0) -> list[Any]:
    bot = max(0, layer - k)
    top = min(layer + k + 1, model.config.num_hidden_layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return [model.transformer.h[l].attn for l in range(bot, top)]
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return [model.gpt_neox.layers[l].attention for l in range(bot, top)]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return [model.model.layers[l].self_attn for l in range(bot, top)]
    raise ValueError(f"Model type {type(model)} not supported")


def norm_module(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "final_layer_norm"):
        return model.gpt_neox.final_layer_norm
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    raise ValueError(f"Model type {type(model)} not supported")


def layer_name(model, num: int, kind: Optional[str] = None) -> str:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
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


def get_module_by_name(model, module_name: str):
    module = model
    for piece in module_name.split("."):
        if piece.isdigit() and hasattr(module, "__getitem__"):
            module = module[int(piece)]
        else:
            module = getattr(module, piece)
    return module


def sublayer_module_name(model, layer_idx: int, target_module: str) -> str:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        mapping = {
            "mlp.fc_in": f"transformer.h.{layer_idx}.mlp.fc_in",
            "mlp.fc_out": f"transformer.h.{layer_idx}.mlp.fc_out",
            "attn.out_proj": f"transformer.h.{layer_idx}.attn.out_proj",
        }
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        mapping = {
            "mlp.fc_in": f"model.layers.{layer_idx}.mlp.up_proj",
            "mlp.fc_out": f"model.layers.{layer_idx}.mlp.down_proj",
            "attn.out_proj": f"model.layers.{layer_idx}.self_attn.o_proj",
        }
    else:
        mapping = {}
    try:
        return mapping[target_module]
    except KeyError as exc:
        raise ValueError(f"Unsupported target_module: {target_module}") from exc


def register_sublayer_hooks(model, target_module: str):
    buffers = {"rep": {}}
    handles = []
    for i in range(model.config.num_hidden_layers):
        module_name = sublayer_module_name(model, i, target_module)
        module = get_module_by_name(model, module_name)

        def make_hook(ii):
            def hook(_module, _inputs, output):
                buffers["rep"][ii] = output.detach().to("cpu")

            return hook

        handles.append(module.register_forward_hook(make_hook(i)))
    return buffers, handles


def find_token_span_by_search(
    tokenizer,
    input_ids,
    substring: str,
    prepend_space: bool = False,
) -> Optional[Tuple[int, int]]:
    candidates = [substring]
    if prepend_space and not substring.startswith(" "):
        candidates.append(" " + substring)
    input_ids = torch.as_tensor(input_ids)
    for candidate in candidates:
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


def _char_span(text: str, substring: str) -> Optional[Tuple[int, int]]:
    start = text.find(substring)
    if start < 0:
        match = re.search(re.escape(substring), text, flags=re.IGNORECASE)
        if not match:
            return None
        return match.span()
    return start, start + len(substring)


def find_token_span_by_offsets(tokenizer, text: str, substring: str) -> Optional[Tuple[int, int]]:
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
    span = _char_span(text, substring)
    if span is None:
        return None
    start_char, end_char = span
    tok_start, tok_end = None, None
    for i, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        if tok_start is None and start < end_char and end > start_char:
            tok_start = i
        if tok_start is not None and start < end_char <= end:
            tok_end = i + 1
            break
        if tok_start is not None and end >= end_char:
            tok_end = i + 1
            break
    if tok_start is None or tok_end is None:
        return None
    return (tok_start, tok_end)


def find_token_range(
    tokenizer,
    text_or_tokens,
    substring: str,
    model_or_name: Any = None,
) -> Optional[Tuple[int, int]]:
    family_source = model_or_name if model_or_name is not None else getattr(tokenizer, "name_or_path", "")
    family = get_model_family(family_source)
    prepend_space = should_prepend_space_for_token_search(family_source)

    if isinstance(text_or_tokens, str):
        text = text_or_tokens
        if family == "qwen3":
            match = find_token_span_by_offsets(tokenizer, text, substring)
            if match is not None:
                return match
        encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded.input_ids[0]
        match = find_token_span_by_search(tokenizer, input_ids, substring, prepend_space=prepend_space)
        if match is not None:
            return match
        return find_token_span_by_offsets(tokenizer, text, substring)

    match = find_token_span_by_search(tokenizer, text_or_tokens, substring, prepend_space=prepend_space)
    if match is not None:
        return match
    tokens = decode_tokens(tokenizer, text_or_tokens)
    whole_string = "".join(tokens)
    char_loc = whole_string.find(substring)
    if char_loc < 0:
        return None
    loc = 0
    tok_start, tok_end = None, None
    for i, token in enumerate(tokens):
        loc += len(token)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    if tok_start is None or tok_end is None:
        return None
    return (tok_start, tok_end)


def model_layers_from_named_modules(model) -> list[str]:
    return [
        name
        for name, _ in model.named_modules()
        if re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", name)
        or re.match(r"^model\.layers\.\d+$", name)
    ]


def path_safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")
