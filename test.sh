# python run_inference.py \
# 	--method pattern_cot \
# 	--dataset aqua \
# 	--demo_path demos/aqua \
# 	--model_name llama \
# 	--model_size 7b \
# 	--model_path PATH_TO_MODEL \
# 	--max_length_cot 4096

# export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_CUMEM_ENABLE=0
# python run_inference_vllm.py \
# 	--method zero_shot_cot \
# 	--dataset strategyqa \
# 	--demo_path demos/strategyqa \
# 	--model_name qwen \
# 	--model_size 7b \
# 	--model_path /root/autodl-tmp/Qwen2.5-7B-Instruct-1M \
# 	--output_dir log/strategyqa \
# 	--max_length_cot 4096 2>&1 >log/strategyqa.log


# python run_inference_vllm.py \
# 	--method zero_shot_cot \
# 	--dataset gsm8k \
# 	--model_name qwen \
# 	--model_size 7b \
# 	--model_path /root/autodl-tmp/Qwen2.5-7B-Instruct-1M \
# 	--output_dir experiment/gsm8k.jsonl \
# 	--max_length_cot 1024 2>&1 >log/GSM8K_zero_shot.log

# python run_inference_vllm.py \
# 	--method zero_shot_cot \
# 	--dataset gsm8k \
# 	--model_name qwen \
# 	--model_size 7b \
# 	--model_path /root/autodl-tmp/llama2_7b_hf \
# 	--output_dir experiment/gsm8k_zero_shot_llama.jsonl \
# 	--max_length_cot 1024 2>&1 >log/GSM8K_zero_shot_llama.log


# python run_inference_vllm.py \
# 	--method zero_shot_cot \
# 	--dataset gsm8k \
# 	--model_name qwen \
# 	--model_size 7b \
# 	--model_path /root/autodl-tmp/Qwen3-4B-Instruct-2507 \
# 	--output_dir experiment/gsm8k_zero_shot_qwen3.jsonl \
# 	--max_length_cot 1024 2>&1 >log/GSM8K_zero_shot_qwen3.log


# Qwen2.5-7B
# Extracted symbols for strategyqa: ['||', 'not', 'the', 'for', 'New', 'you', '!=', 'Zealand.', '÷', '∩', '>=', 'decision-making', 'as', '≈', 'comparison', 'why', '⊗', '10', '∨', 'appropriate.', 'is', 'instruction.', 'involves', '?', 'list', 'are', '&', '⊕', '并集', 'operations', 'Explanation:', 'used', 'conditions', '⊃', 'and', 'consider', 'evaluating', 'per', 'a', 'These', 'question', 'in', 'various', 'operators', 'spades', 'cups', 'diamonds', '⊆', 'which', '⊂', 'could', 'hearts', 'to', 'logical', 'whether', 'multiple', 'kayak', 'The', '交集', 'OR', '==', '∧', 'limited', 'represent', '&&', 'using', 'processes', 'symbols', 'traditional', 'reasoning', '∈', ':.', 'also', 'However', '×', 'Answer:', 'or', '<=', '∪', '⊇', 'similar', '|']
# Qwen3-235B
# Extracted symbols for strategyqa: ['≠', '=', '∈', '∃', '∀', '∉', '∩', '?', '⊆', '∪']
# python run_demo.py \
# 	--task strategyqa \
# 	--pred_file log/strategyqa.log \
# 	--demo_save_dir demos/strategyqa \
# 	--encoder /root/autodl-tmp/jina-embeddings-v2-base-code


# Extracted symbols for gsm8k: ['the', 'relevant', '//', 'etc.', 'on', '%=', '*', '+', 'calculating', 'based', 'For', 'of', '%', 'task', '#operators', 'are:', '/', '///', '#', '#operator*', 'operators', 'total', '=.', '-', '>', '=', 'rates', 'given', 'cost']
# python run_demo.py \
# 	--task gsm8k \
# 	--pred_file log/gsm8k.log \
# 	--demo_save_dir demos/GSM8K \
# 	--encoder /root/autodl-tmp/jina-embeddings-v2-base-code


# python run_inference_vllm.py \
# 	--method pattern_cot \
# 	--dataset strategyqa \
# 	--demo_path demos/strategyqa \
# 	--model_name qwen \
# 	--model_size 7b \
# 	--model_path /root/autodl-tmp/Qwen2.5-7B-Instruct-1M \
# 	--max_length_cot 4096


# nohup python run_inference_vllm.py \
# 	--method mcu_cot \
# 	--dataset gsm8k \
# 	--demo_path gsm8k_parsed.json \
# 	--model_name qwen \
# 	--model_size 7b \
# 	--model_path /root/autodl-tmp/Qwen2.5-7B-Instruct-1M \
# 	--output_dir experiment/gsm8k_em_example.jsonl \
# 	--max_length_cot 1024 2>&1 >log/GSM8K_em_example.log

# python run_inference_vllm.py \
# 	--method pattern_cot \
# 	--dataset gsm8k \
# 	--demo_path demos/gsm8k \
# 	--model_name qwen \
# 	--model_size 7b \
# 	--model_path /root/autodl-tmp/Qwen2.5-7B-Instruct-1M \
# 	--output_dir experiment/gsm8k_pattern_example.jsonl \
# 	--max_length_cot 1024 2>&1 >log/GSM8K_pattern_example.log

#----------------------------------------------------------------------------------


# python run_inference_vllm.py \
# 	--method zero_shot \
# 	--dataset gsm8k \
# 	--demo_path demos/gsm8k \
# 	--model_name llama \
# 	--model_size 7b \
# 	--model_path /root/autodl-tmp/llama2_7b_hf \
# 	--output_dir experiment/gsm8k_zero_shot_llama7b.jsonl \
# 	--max_length_cot 512 2>&1 >log/GSM8K_zero_shot_llama7b.log


# python run_inference_vllm.py \
# 	--method pattern_cot \
# 	--dataset gsm8k \
# 	--demo_path demos/gsm8k \
# 	--model_name llama \
# 	--model_size 7b \
# 	--model_path /root/autodl-tmp/llama2_7b_hf \
# 	--output_dir experiment/gsm8k_pattern_example_llama7b.jsonl \
# 	--max_length_cot 1024 2>&1 >log/GSM8K_pattern_example_llama7b.log


# python run_inference_vllm.py \
# 	--method mcu_cot \
# 	--dataset gsm8k \
# 	--demo_path gsm8k_llama7b_parsed.json \
# 	--model_name llama \
# 	--model_size 7b \
# 	--model_path /root/autodl-tmp/llama2_7b_hf \
# 	--output_dir experiment/gsm8k_gold_example_llama7b.jsonl \
# 	--max_length_cot 1024 2>&1 >log/GSM8K_gold_example_llama7b.log

#----------------------------------------------------------------------------------

python run_inference_vllm.py \
	--method zero_shot \
	--dataset gsm8k \
	--demo_path demos/gsm8k \
	--model_name qwen \
	--model_size 7b \
	--model_path /root/autodl-tmp/Qwen2.5-7B-Instruct \
	--output_dir experiment/gsm8k_zero_shot_qwen7b.jsonl \
	--max_length_cot 512 2>&1 >log/GSM8K_zero_shot_qwen7b.log


python run_inference_vllm.py \
	--method pattern_cot \
	--dataset gsm8k \
	--demo_path demos/gsm8k \
	--model_name qwen \
	--model_size 7b \
	--model_path /root/autodl-tmp/Qwen2.5-7B-Instruct \
	--output_dir experiment/gsm8k_pattern_example_qwen7b.jsonl \
	--max_length_cot 1024 2>&1 >log/GSM8K_pattern_example_qwen7b.log


python run_inference_vllm.py \
	--method mcu_cot \
	--dataset gsm8k \
	--demo_path gsm8k_qwen7b_parsed.json \
	--model_name qwen \
	--model_size 7b \
	--model_path /root/autodl-tmp/Qwen2.5-7B-Instruct \
	--output_dir experiment/gsm8k_gold_example_qwen7b.jsonl \
	--max_length_cot 1024 2>&1 >log/GSM8K_gold_example_qwen7b.log

#----------------------------------------------------------------------------------

# python run_inference_vllm.py \
# 	--method zero_shot \
# 	--dataset gsm8k \
# 	--demo_path demos/gsm8k \
# 	--model_name qwen \
# 	--model_size 3b \
# 	--model_path /root/autodl-tmp/Qwen2.5-3B-Instruct \
# 	--output_dir experiment/gsm8k_zero_shot_qwen3b.jsonl \
# 	--max_length_cot 512 2>&1 >log/GSM8K_zero_shot_qwen3b.log

# python run_inference_vllm_str.py \
# 	--method zero_shot \
# 	--dataset gsm8k \
# 	--demo_path demos/gsm8k \
# 	--model_name qwen \
# 	--model_size 3b \
# 	--model_path /root/autodl-tmp/Qwen2.5-3B-Instruct \
# 	--output_dir experiment/gsm8k_zero_shot_qwen3b.jsonl \
# 	--max_length_cot 512 2>&1 >log/GSM8K_zero_shot_qwen3b.log


# python run_inference_vllm.py \
# 	--method pattern_cot \
# 	--dataset gsm8k \
# 	--demo_path demos/gsm8k \
# 	--model_name qwen \
# 	--model_size 3b \
# 	--model_path /root/autodl-tmp/Qwen2.5-3B-Instruct \
# 	--output_dir experiment/gsm8k_pattern_example_qwen3b.jsonl \
# 	--max_length_cot 1024 2>&1 >log/GSM8K_pattern_example_qwen3b.log


# python run_inference_vllm.py \
# 	--method mcu_cot \
# 	--dataset gsm8k \
# 	--demo_path gsm8k_qwen3b_parsed.json \
# 	--model_name qwen \
# 	--model_size 3b \
# 	--model_path /root/autodl-tmp/Qwen2.5-3B-Instruct \
# 	--output_dir gsm8k_gold_example_qwen3b.jsonl \
# 	--max_length_cot 1024 2>&1 >log/GSM8K_gold_example_qwen3b.log

#----------------------------------------------------------------------------------

# python run_inference_vllm.py \
# 	--method zero_shot_cot \
# 	--dataset gsm8k \
# 	--demo_path demos/gsm8k \
# 	--model_name llama \
# 	--model_size 13b \
# 	--model_path /root/autodl-tmp/llama2_13b_hf \
# 	--output_dir experiment/gsm8k_zero_shot_llama13b.jsonl \
# 	--max_length_cot 512 2>&1 >log/GSM8K_zero_shot_llama13b.log


# python run_inference_vllm.py \
# 	--method pattern_cot \
# 	--dataset gsm8k \
# 	--demo_path demos/gsm8k \
# 	--model_name llama \
# 	--model_size 13b \
# 	--model_path /root/autodl-tmp/llama2_13b_hf \
# 	--output_dir experiment/gsm8k_pattern_example_llama13b.jsonl \
# 	--max_length_cot 1024 2>&1 >log/GSM8K_pattern_example_llama13b.log

# python run_inference_vllm.py \
# 	--method mcu_cot \
# 	--dataset gsm8k \
# 	--demo_path gsm8k_llama13b_parsed.json \
# 	--model_name llama \
# 	--model_size 13b \
# 	--model_path /root/autodl-tmp/llama2_13b_hf \
# 	--output_dir experiment/gsm8k_gold_example_llama13b.jsonl \
# 	--max_length_cot 1024 2>&1 >log/GSM8K_gold_example_llama13b.log