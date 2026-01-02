export CUDA_VISIBLE_DEVICES=2

# python model_structure.py

# python classify_model_performance.py \
#     --data /data/xkliu/Knowledge_Edit/AlphaEdit/data/MQuAKE-CF-3k-v2.json \
#     --model /data/share_weight/Qwen2.5-7B-Instruct \
#     --output_dir ./Qwen2.5-7B_classified_split_results/ \
#     --mode separate \

python analyze_same_layer_sim.py \
  --data ./Llama-3-8B_classified_split_results/class_1_both_correct.json \
  --model /data/share_weight/Meta-Llama-3-8B-Instruct \
  --target_module mlp.fc_in \
  --out_dir ./outputs/Llama-3-8B_classified_split_results/analyze_same_layer_sim/class_1_both_correct/mlp_fc_in \
  --norm zscore \
  --cpu_threads 10

# python filter_valid_entity_row.py \
#   --input Llama-3-8B_classified_split_results/class_4_both_wrong.json \
#   --output Llama-3-8B_classified_split_results_filter/class_4_both_wrong_valid.json \
#   --model /data/share_weight/Meta-Llama-3-8B-Instruct \

# /data/share_weight/Meta-Llama-3-8B-Instruct
# /data/xkliu/hf_models/gpt-j-6b
# mlp.fc_in mlp.fc_out attn.out_proj
