python code/patchscopes/analyze_generation_results.py \
    --json Llama-3-8B_classified_split_results/class_4_both_wrong.json \
    --input output/gpt-j-6B_classified_split_results_filter/description/last/run_0/class_4_both_wrong_caseFilter90.jsonl output/gpt-j-6B_classified_split_results_filter/description/last/run_1/class_4_both_wrong_caseFilter90.jsonl output/gpt-j-6B_classified_split_results_filter/description/last/run_2/class_4_both_wrong_caseFilter90.jsonl\
    --output results/gpt-j-6B_classified_split_results_filter/last/class_4_both_wrong/class_4_both_wrong_caseFilter90.jsonl \
    --min_layer -1