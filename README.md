初期实验在 GSM8K 数据集上测试了三种提示词方法：  
- Zero-Shot CoT  
- Pattern-CoT  
- MCU-CoT  

测试脚本如下：

### 1. Zero-Shot CoT
```bash
python run_inference_vllm.py \
  --method zero_shot \
  --dataset gsm8k \
  --demo_path demos/gsm8k \
  --model_name qwen \
  --model_size 7b \
  --model_path /root/autodl-tmp/Qwen2.5-7B-Instruct \
  --output_dir experiment/gsm8k_zero_shot_qwen7b.jsonl \
  --max_length_cot 512 2>&1 >log/GSM8K_zero_shot_qwen7b.log
```

### 2. Pattern-CoT
```bash
python run_inference_vllm.py \
  --method pattern_cot \
  --dataset gsm8k \
  --demo_path demos/gsm8k \
  --model_name qwen \
  --model_size 7b \
  --model_path /root/autodl-tmp/Qwen2.5-7B-Instruct \
  --output_dir experiment/gsm8k_pattern_example_qwen7b.jsonl \
  --max_length_cot 1024 2>&1 >log/GSM8K_pattern_example_qwen7b.log
```

### 3. MCU-CoT
```bash
python run_inference_vllm.py \
  --method mcu_cot \
  --dataset gsm8k \
  --demo_path gsm8k_qwen7b_parsed.json \
  --model_name qwen \
  --model_size 7b \
  --model_path /root/autodl-tmp/Qwen2.5-7B-Instruct \
  --output_dir experiment/gsm8k_gold_example_qwen7b.jsonl \
  --max_length_cot 1024 2>&1 >log/GSM8K_gold_example_qwen7b.log
```

> **注意：**  
> 运行 Pattern-CoT 前，需先对 Zero-Shot CoT 的结果进行预处理，完成运算 token 提取、去重与分类（生成 `gsm8k_qwen7b_parsed.json` 文件），以提取 examples。  
> 目前该部分代码尚未上传。  
> Pattern-CoT 实际依赖预定义的运算 token 集合，通过 `run_demo.py` 脚本获取相应 examples。

可使用 `view.ipynb` 脚本对比查看两种提示词方法在特定题目上的表现差异。