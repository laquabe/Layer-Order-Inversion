python experiments/context_expansion.py \
  --input_file data/llama3-8b-c3_2hop.json \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --output_dir results/llama3-8b-c3_2hop/ \
  --local \
  --context_mode filler \
  --check_single_hops \
  --max_new_tokens 128 \
  --overwrite