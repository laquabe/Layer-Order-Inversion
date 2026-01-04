export CUDA_VISIBLE_DEVICES=1
python code/patchscope/generate_entity_description.py /data/share_weight/Meta-Llama-3-8B-Instruct last \
  --input-path ./datasets/Llama-3-8B_classified_split_results_sample/class_3_single_correct_multi_wrong_valid.csv \
  --output-path ./output/Llama-3-8B_classified_split_results_sample/description/last/run_2/class_3_single_correct_multi_wrong_valid.jsonl \
  --target-prompt description \
  --batch-size 128 \
  --seed 0