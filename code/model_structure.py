from transformers import AutoModelForCausalLM

MODEL_NAME = "/data/share_weight/Meta-Llama-3-8B-Instruct"  # 可换成 "meta-llama/Llama-2-7b-hf"、"mistralai/Mistral-7B-v0.1" 等

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print(f"=== 模型结构: {MODEL_NAME} ===\n")

for name, module in model.named_modules():
    print(name, "->", module.__class__.__name__)
