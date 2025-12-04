"""

extract operators from GSM8K rationales

"""


#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List
from collections import defaultdict

import tqdm
from openai import OpenAI

# INPUT_JSON      = "gsm8k_goldens.json"
# INPUT_JSON      = 'gsm8k_qwen7b.json'
# INPUT_JSON      = 'gsm8k_qwen3b.json'
# INPUT_JSON      = 'gsm8k_llama7b.json'
INPUT_JSON      = 'gsm8k_llama13b.json'

OUTPUT_JSON     = f'{INPUT_JSON.split(".json")[0]}_parsed.json'
OPS_CACHE_JSONL = f'{INPUT_JSON.split(".json")[0]}_ops_cache.json'
BATCH_FLUSH     = 50
# --------------------------

# ---------- LLM ----------
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "your_api_key_here"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)

SYS_PROMPT = (
    "You are an assistant that extracts arithmetic operators from a reasoning text. "
    "The text may contain LaTeX macros like \\times, \\div, \\frac, or plain symbols like *, /, ÷. "
    "Rules:\n"
    "1. Return the operators in the order they are **first used to compute a numeric value**.\n"
    "2. Exclude equality/assignment symbols (=, \\eq, \\Rightarrow, etc.) — they are not arithmetic operators.\n"
    "3. Map all variants to the **unified symbols** below:\n"
    "   - addition: always use '+'\n"
    "   - subtraction: always use '-'\n"
    "   - multiplication: always use '*' (map \\times and * to *)\n"
    "   - division: always use '/' (map \\div, /, and ÷ to /)\n"
    "4. Output only a JSON list of strings, e.g. ['+', '*', '-', '/']"
)

def extract_ops(rationale: str) -> List[str]:
    rationale = rationale.replace("\n", " ")
    resp = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": rationale},
        ],
        max_tokens=128,
        temperature=0.0,
    )
    content = resp.choices[0].message.content.strip()
    try:
        ops = json.loads(content)
        if isinstance(ops, list) and all(isinstance(o, str) for o in ops):
            return ops
    except Exception:
        pass
    return []


def main() -> None:
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        raw_records = json.load(f)["demo"]

    done = set()
    cache_file = Path(OPS_CACHE_JSONL)
    if cache_file.exists():
        with cache_file.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["id"])
        print(f"已处理 {len(done)} 条，继续...")

    writer = cache_file.open("a", encoding="utf-8")
    try:
        for idx, rec in enumerate(tqdm.tqdm(raw_records, desc="LLM extract ops")):
            if idx in done:
                continue
            ops = extract_ops(rec.get("rationale", ""))
            writer.write(json.dumps({"id": idx, "operators": ops}, ensure_ascii=False) + "\n")
            if (idx + 1) % BATCH_FLUSH == 0:
                writer.flush()
    finally:
        writer.close()

    id2ops = {}
    with cache_file.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                id2ops[obj["id"]] = obj["operators"]

    demo_list = []
    for idx, rec in enumerate(raw_records):
        demo_list.append({
            "id": idx,
            "question": rec.get("question"),
            "rationale": rec.get("rationale", "").replace("\n", " "),
            "operators": id2ops[idx],
            "pred_ans": rec.get("pred_ans"),
            "gold_ans": rec.get("gold_ans"),
        })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"demo": demo_list}, f, indent=4, ensure_ascii=False)
    print(f"✅ 全部完成 → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()



"""

lookup same operators

"""

id2ops = {}
with open(OPS_CACHE_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            id2ops[obj["id"]] = obj["operators"]

with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)
demo = data["demo"]


for item in demo:
    item["operators"] = id2ops[item["id"]]

op2ids = defaultdict(list)
for item in demo:
    op2ids[tuple(item["operators"])].append(item["id"])


lookup = {str(ops): ids for ops, ids in op2ids.items()}

for item in demo:
    ops = tuple(item["operators"])
    same = [i for i in op2ids[ops] if i != item["id"]]
    item["same_ops_ids"] = same

data["lookup"] = lookup
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ 完成：共 %d 种 operators，已更新 same_ops_ids 与 lookup" % len(lookup))
