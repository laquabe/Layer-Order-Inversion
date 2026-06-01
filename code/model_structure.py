import argparse

from model_support import load_causal_lm, load_tokenizer, resolve_model_path


parser = argparse.ArgumentParser(description="Print model module and tokenizer structure.")
parser.add_argument(
    "--model_name",
    default="/data/share_weight/Meta-Llama-3-8B-Instruct",
    help="Local path or HF model name.",
)
parser.add_argument("--local", action="store_true")
args = parser.parse_args()

model_path = resolve_model_path(args.model_name, local=args.local)
model = load_causal_lm(args.model_name, local=args.local)
tokenizer = load_tokenizer(args.model_name, local=args.local)

print(f"=== 模型结构: {args.model_name} ({model_path}) ===\n")

for name, module in model.named_modules():
    print(name, "->", module.__class__.__name__)

# ==== 关键测试部分 ====
print("=== Tokenizer 信息 ===")
print(type(tokenizer))
print("name_or_path:", tokenizer.name_or_path)

text = "Paris"

ids_with_special = tokenizer.encode(text)
ids_without_special = tokenizer(text, add_special_tokens=False)["input_ids"]

print("\n=== 编码对比 ===")
print("with special tokens:", ids_with_special)
print("without special tokens:", ids_without_special)

# 对应 token 字符串
print("\n=== decode 对比 ===")
print("with special:", tokenizer.convert_ids_to_tokens(ids_with_special))
print("without special:", tokenizer.convert_ids_to_tokens(ids_without_special))

# ==== 检查 BOS 是否存在 ====
if ids_with_special[0] != ids_without_special[0]:
    print("\n[检测] 很可能存在 BOS（或其他 special token）差异")
    print("差值:", len(ids_with_special) - len(ids_without_special))
else:
    print("\n[检测] 没有发现 BOS 差异（至少在这个样例上）")

