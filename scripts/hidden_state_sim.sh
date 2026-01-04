python analyze_same_layer_sim.py \
  --data ./Llama-3-8B_classified_split_results/class_1_both_correct.json \
  --model /data/share_weight/Meta-Llama-3-8B-Instruct \
  --target_module mlp.fc_in \
  --out_dir ./outputs/Llama-3-8B_classified_split_results/analyze_same_layer_sim/class_1_both_correct/mlp_fc_in \
  --norm zscore \
  --cpu_threads 10