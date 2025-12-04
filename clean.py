#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import tqdm
from openai import OpenAI

ENABLE_CLEAN: bool = True
BATCH_DUMP: int = 50          #flush

# ---------- LLM ----------
try:
    CLIENT = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "your_api_key_here"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
except ImportError:
    if ENABLE_CLEAN:
        raise RuntimeError("ENABLE_CLEAN=True 需 pip install openai")

SYS_PROMPT = (
    "You are an assistant that cleans a reasoning field.\n"
    "Please rewrite the following rationale so that:\n"
    "1. It MUST start with \"Let's think step by step. \" (include the trailing space);\n"
    "2. It MUST end with \"The answer (Arabic numerals) is <number>\" (where <number> is the final numeric answer);\n"
    "3. Remove any heading like \"Explanation\", \"Answer\", \"Step-by-step\" etc.\n"
    "4. Keep the logical steps concise and correct."
)
ANS_RE = re.compile(r"The answer \(Arabic numerals\) is (\d+)")


def clean_rationale(text: str) -> Tuple[str, str]:
    """return (cleaned_rationale, extracted_number)"""
    resp = CLIENT.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": text},
        ],
        max_tokens=1024,
        temperature=0.0,
    )
    cleaned = resp.choices[0].message.content.strip()
    if not cleaned.startswith("Let's think step by step. "):
        cleaned = "Let's think step by step. " + cleaned
    m = ANS_RE.search(cleaned)
    number = m.group(1) if m else ""
    cleaned = re.sub(r"The answer.*$", "", cleaned).rstrip() + f" The answer (Arabic numerals) is {number}"
    return cleaned, number


def main(
    src_jsonl: Path = Path("experiment/gsm8k_gold_example_llama.jsonl"),
    dst_jsonl: Path = Path("experiment/gsm8k_gold_example_llama_clean.jsonl"),
) -> None:
    dst_jsonl.parent.mkdir(parents=True, exist_ok=True)

    already = 0
    if dst_jsonl.exists():
        with dst_jsonl.open(encoding="utf-8") as f:
            already = sum(1 for _ in f if _.strip())
        print(f"检测到已写入 {already} 行，继续追加...")

    with src_jsonl.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    total = len(records)
    if already >= total:
        print("所有记录已处理完毕，无需重复运行")
        return

    keys = {"question", "gold_ans", "rationale", "pred_ans", "wrap_que"}
    fail_cnt = 0
    with dst_jsonl.open("a", encoding="utf-8") as fo:
        for idx in tqdm.trange(already, total, desc="Cleaning"):
            obj = records[idx]
            rationale = obj.get("rationale", "")
            pred_ans = obj.get("pred_ans", "")

            if ENABLE_CLEAN and rationale:
                rationale, extracted = clean_rationale(rationale)
                if extracted:
                    pred_ans = extracted
                else:
                    fail_cnt += 1

            out = {k: obj.get(k, "") for k in keys}
            out.update({"rationale": rationale, "pred_ans": pred_ans})
            json.dump(out, fo, ensure_ascii=False)
            fo.write("\n")

            # 定期 flush
            if (idx + 1) % BATCH_DUMP == 0:
                fo.flush()
                print(f"已处理 {idx + 1}/{total}，提取失败 {fail_cnt}")

    print(f"全部完成 → {dst_jsonl}，总计 {total} 条，提取失败 {fail_cnt}")


if __name__ == "__main__":
    # SRC_JSONL = Path("experiment/gsm8k_gold_example_llama.jsonl")
    # DST_JSONL = Path("experiment/gsm8k_gold_example_llama_clean.jsonl")

    # SRC_JSONL = Path("experiment/gsm8k_zero_shot_llama.jsonl")
    # DST_JSONL = Path("experiment/gsm8k_zero_shot_llama_clean.jsonl")

    # SRC_JSONL = Path("experiment/gsm8k_gold_example_qwen3b.jsonl")
    # DST_JSONL = Path("experiment/gsm8k_gold_example_qwen3b_clean.jsonl")

    # SRC_JSONL = Path("experiment/gsm8k_zero_shot_qwen3b.jsonl")
    # DST_JSONL = Path("experiment/gsm8k_zero_shot_qwen3b_clean.jsonl")

    # SRC_JSONL = Path('experiment/gsm8k_zero_shot_qwen7b.jsonl')
    # DST_JSONL = Path('experiment/gsm8k_zero_shot_qwen7b_clean.jsonl')

    SRC_JSONL = Path("experiment/gsm8k_zero_shot_llama13b.jsonl")
    DST_JSONL = Path("experiment/gsm8k_zero_shot_llama13b_clean.jsonl")
    main(SRC_JSONL, DST_JSONL)
