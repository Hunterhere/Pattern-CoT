import argparse
import json
import numpy as np
from collections import Counter, defaultdict
from utils import *

def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    fix_seed(args.random_seed)

    decoder = Decoder(args.model_name, args.model_size, args.model_path)
    dataloader = setup_data_loader(args)
    print_now()

    if args.method == "few_shot":
        demo = create_demo_text(args, cot_flag=False)
    elif args.method in ["few_shot_cot", "auto_cot", "pattern_cot"]:
        demo = create_demo_text(args, cot_flag=True)
    elif args.method == "test_cot":
        demo = ""
    else:
        demo = ""

    batch_size = 16  #TODO
    batch_prompts = []
    batch_meta = []

    total = 0
    correct_total = 0

    with open(args.output_dir, "a") as wp:
        for i, data in enumerate(dataloader):
            if i < args.resume_id - 1:
                continue

            x, y = data
            x = "Q: " + x[0] + "\n" + "A:"
            y = y[0].strip()

            if args.method == "zero_shot":
                prompt = x + " " + args.direct_answer_trigger_for_zeroshot
            elif args.method == "zero_shot_cot":
                prompt = x + " " + args.cot_trigger
            elif args.method in ["few_shot", "few_shot_cot"]:
                prompt = demo + x
            elif args.method in ["auto_cot", "pattern_cot"]:
                prompt = "Given Some Examlpes you can learn from and answer the following Question.\nExamples: \n####\n" + demo + "####\n Only Answer this Question, End with the answer number: \n" + x + " " + args.cot_trigger
                # print(prompt)
            elif args.method == "test_cot":
                with open(args.demo_path, encoding="utf-8") as f:
                    demo_pool = json.load(f)["demo"]
                same_ids = demo_pool[i]["same_ops_ids"]

                #FIXME: 按字符串长度升序排序，优先选短的示例
                candidate_demos = [demo_pool[_id] for _id in same_ids]
                candidate_demos.sort(key=lambda ex: len(ex["question"] + " " + ex["rationale"]))
                demo_ex = candidate_demos[:3]

                demo_text = "".join(
                    ex["question"] + " " + ex["rationale"] + ".\n\n"
                    for ex in demo_ex
                )
                prompt = "Given Some Examlpes you can learn from and answer the following Question.\nExamples: \n####\n" + demo_text + "####\n Only Answer this Question, End with the answer number: \n" + x + " " + args.cot_trigger
            else:
                raise ValueError("method is not properly defined ...")

            max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct

            for _ in range(args.iterations):
                batch_prompts.append(prompt)
                batch_meta.append({
                    "idx": i,
                    "gold_ans": y,
                    "question": x,
                    "prompt": prompt,
                    "max_length": max_length
                })

            if len(batch_prompts) >= batch_size:
                process_batch(decoder, batch_prompts, batch_meta, wp, args, correct_total, total)
                batch_prompts.clear()
                batch_meta.clear()

            if args.limit_dataset_size and (i + 1) >= args.limit_dataset_size:
                break

        if batch_prompts:
            correct_total, total = process_batch(decoder, batch_prompts, batch_meta, wp, args, correct_total, total)

    accuracy = (correct_total * 1.0 / total) * 100
    print("Final accuracy : {:.2f}%".format(accuracy))
    convert_gsm8k_jsonl_to_json(args.output_dir, f"{args.dataset}_{args.model_name}{args.model_size}_results.json")


def process_batch(decoder, batch_prompts, batch_meta, wp, args, correct_total, total):
    responses = decoder.decode(batch_prompts, args.model_name, batch_meta[0]["max_length"])
    print("RAW RESPONSE: ", responses)

    grouped = defaultdict(list)
    for meta, resp in zip(batch_meta, responses):
        idx = meta["idx"]
        grouped[idx].append({
            "resp": resp,
            "gold": meta["gold_ans"],
            "question": meta["question"],
            "prompt": meta["prompt"]
        })

    for idx, items in grouped.items():
        preds = []
        rationales = []
        for item in items:
            resp = item["resp"]
            if args.method == "zero_shot_cot":
                z2 = item["prompt"] + resp + " " + args.direct_answer_trigger_for_zeroshot_cot
                pred = decoder.decode([z2], args.model_name, args.max_length_direct)[0]
            else:
                pred = resp
            cleaned = answer_cleansing(args, pred)
            preds.append(cleaned)
            rationales.append(resp)

        final_pred = Counter(preds).most_common(1)[0][0]
        gold = items[0]["gold"]

        output_line = {
            "question": items[0]["question"],
            "gold_ans": gold,
            "pred_ans": final_pred,
            "rationale": "\n".join(rationales),
            "wrap_que": items[0]["prompt"]
        }

        wp.write(json.dumps(output_line) + '\n')

        correct = (np.array([final_pred]) == np.array([gold])).sum().item()
        correct_total += correct
        total += 1
        print(f"[{total}] pred: {final_pred}, gold: {gold}, acc: {correct_total/total*100:.2f}%")

    return correct_total, total

def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="addsub", choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith",  "strategyqa", "svamp", "singleeq", "coin_flip", "last_letters", "object_tracking", "bigbench_date"], help="dataset used for experiment"
    )
    parser.add_argument(
        "--demo_path", type=str, default="demos/addsub", help="pre-generated demos used for experiment"
    )
    parser.add_argument(
        "--resume_id", type=int, default=0, help="resume from which question id (current line number in the output file), if the experiment fails accidently (e.g., network error)"
    )
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")
    
    parser.add_argument("--max_num_worker", type=int, default=0, help="maximum number of workers for dataloader")
    
    parser.add_argument(
        "--method", type=str, default="zero_shot_cot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "auto_cot", "pattern_cot","test_cot"], help="method"
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiment/addsub", help="output directory"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=256, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=256, help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument( 
        "--limit_dataset_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="temperature"
    )
    parser.add_argument(
        "--iterations", type=int, default=1, help="self consistency iterations"
    )
    parser.add_argument(
        "--model_name", type=str, default="qwen", help="model used for decoding."
    )
    parser.add_argument(
        "--model_size", type=str, default="7b", help="model size"
    )
    parser.add_argument(
        "--model_path", type=str, default="/root/autodl-tmp/Qwen2.5-7B-Instruct-1M", help="path to the model. Use Hugging Face ID for vLLM."
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is option "
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is "
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is "
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is option "
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is "
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the (Yes or No) answer is: "
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is "
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step. "
    
    return args

if __name__ == "__main__":
    main()
    