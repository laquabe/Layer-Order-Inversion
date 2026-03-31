python code/latent_patch/experiments/last_hop_repair.py \
  --input_file code/latent_patch/data/llama3-8b-c3_2hop.json \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --output_dir results/llama3-8b-c3_2hop \
  --local \
  --last_k 3 \
  --max_new_tokens 128 \
  --overwrite
