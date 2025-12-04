from statistics import mean
from torch.utils.data import Dataset
import os
import multiprocessing
import json
import numpy as np
import torch
import re
import random
import time
import datetime
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from vllm import LLM, SamplingParams
from vllm.sampling_params import SamplingParams, GuidedDecodingParams
from typing import List
from pathlib import Path

def shuffleDict(d):
    keys = list(d.keys())
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    keys = [(key, d[key]) for key in keys]
    #keys = d(keys)
    return dict(keys)
  
def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    #pass
    
def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    now = now.strftime('%Y/%m/%d %H:%M:%S')
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass


class Decoder: # Batch version
    def __init__(self, model_name, size, path):
        self.sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=1024, repetition_penalty=1.2)
        self.model = LLM(
            model=path,
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype="bfloat16",
            gpu_memory_utilization=0.95,
            max_num_seqs=128,
            # max_model_len=8192
        )

    def decode(self, prompts, model_name, max_length=2048):
        self.sampling_params.max_tokens = max_length
        outputs = self.model.generate(prompts, self.sampling_params, use_tqdm=False)
        return [out.outputs[0].text for out in outputs]
    
    def decode_structured(self,
                          prompts: List[str],
                          max_length: int,
                          schema: type) -> List[str]:
        
        guided = GuidedDecodingParams(json=schema.model_json_schema())
        sp = SamplingParams(
            max_tokens=max_length,
            temperature=0.0,
            guided_decoding=guided
        )
        outs = self.llm.generate(prompts, sp)
        return [o.outputs[0].text for o in outs]

def data_reader(args):

    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset == "aqua":
      with open(args.dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          choice = "(" + "(".join(json_res["options"])
          choice = choice.replace("(", " (").replace(")", ") ")
          choice = "Answer Choices:" + choice
          questions.append(json_res["question"].strip() + " " + choice)
          answers.append(json_res["correct"])
  
    elif args.dataset == "gsm8k":
      with open(args.dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          questions.append(json_res["question"].strip())
          answers.append(json_res["answer"].split("#### ")[-1])
  
    elif args.dataset == "commonsensqa":
      with open(args.dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          choice = "Answer Choices:"
          for c in json_res["question"]["choices"]:
              choice += " ("
              choice += c["label"]
              choice += ") "
              choice += c["text"]
          questions.append(json_res["question"]["stem"].strip() + " " + choice)
          answers.append(json_res["answerKey"])

    elif args.dataset in ("addsub", "multiarith", "singleeq"):
      with open(args.dataset_path) as f:
        json_data = json.load(f)
        for line in json_data:
          q = line["sQuestion"].strip()
          a = str(line["lSolutions"][0])
          if a[-2:] == ".0":
              a = a[:-2]
          questions.append(q)
          answers.append(a)
        
    elif args.dataset == "strategyqa":
      with open(args.dataset_path) as f:
        json_data = json.load(f)["examples"]
        for line in json_data:
          q = line["input"].strip()
          a = int(line["target_scores"]["Yes"])
          if a == 1:
              a = "yes"
          else:
              a = "no"
          questions.append(q)
          answers.append(a)
        
    elif args.dataset == "svamp":
      with open(args.dataset_path) as f:
        json_data = json.load(f)
        for line in json_data:
            q = line["Body"].strip() + " " + line["Question"].strip()
            a = str(line["Answer"])
            if a[-2:] == ".0":
                a = a[:-2]
            questions.append(q)
            answers.append(a)
            
    elif args.dataset in ("bigbench_date", "object_tracking"):
      with open(args.dataset_path) as f:
        json_data = json.load(f)
        json_data = json_data["examples"]
        if args.dataset == "bigbench_date":
            choice_index = ['A','B','C','D','E','F']
        elif args.dataset in ("object_tracking"):
            choice_index = ['A','B','C']
        else:
            raise ValueError("dataset is not properly defined ...")
        for line in json_data:
          q = line["input"].strip()
          if args.dataset == "bigbench_date":
              choice = "Answer Choices:"
              # Randomly shuffle the answer choice dictionary because the original answer is always A ...
              choice_dic = shuffleDict(line["target_scores"])
          elif args.dataset == "object_tracking":
              choice = "\nWhich choice is true ? Answer Choices:"
              choice_dic = line["target_scores"]
          else:
              raise ValueError("dataset is not properly defined ...")
          for i, key_value in enumerate(choice_dic.items()):
              key, value = key_value
              choice += " ("
              choice += choice_index[i]
              choice += ") "
              choice += key
              if value == 1:
                  a = choice_index[i]
          q = q + " " + choice
          questions.append(q)
          answers.append(a)                     

    elif args.dataset in ("coin_flip", "last_letters"):
      with open(args.dataset_path) as f:
        json_data = json.load(f)
        json_data = json_data["examples"]
        for line in json_data:
          q = line["question"]
          a = line["answer"]
          questions.append(q)
          answers.append(a)
        
    else:
        raise ValueError("dataset is not properly defined ...")
    
    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)
    
    print("dataset : {}".format(args.dataset))
    print("data size : {}".format(len(answers)))
    print("average num of words for each sample : {}".format(q_len_mean))
    
    return questions, answers

# Create dataset object before dataloader ...
class MyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.questions, self.answers = data_reader(args)
        self.len = len(self.questions)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        input = self.questions[index]
        output = self.answers[index]
        return input, output

def setup_data_loader(args):

    # fix randomness of dataloader to ensure reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    fix_seed(args.random_seed)
    worker_seed = torch.initial_seed() % 2**32
    print("worker_seed : {}".format(worker_seed))
    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(worker_seed)
    
    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, args.max_num_worker)
    print("dataloader_num_workers: " + str(dataloader_num_workers))
    
    dataset = MyDataset(args)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                  shuffle=True,
                  batch_size=args.minibatch_size,
                  drop_last=False,
                  num_workers=dataloader_num_workers,
                  worker_init_fn=seed_worker,
                  generator=g,
                  pin_memory=True)

    return dataloader

def answer_cleansing(args, pred, must_choice=False):
    if not hasattr(answer_cleansing, 'fail_count'):
        answer_cleansing.fail_count = 0

    original_pred = pred
    print("pred_before : " + pred)

    triggers = [
        args.direct_answer_trigger_for_fewshot.lower(),  # “the answer is”
        "result is",
        "final answer is"
    ]
    answer_flag = False
    pred_lower = pred.lower()
    for t in triggers:
        if t in pred_lower:
            pred = pred[pred_lower.rfind(t) + len(t):]
            answer_flag = True
            break

    if args.dataset in ("aqua", "commonsensqa"):
        pred = re.findall(r'A|B|C|D|E', pred)
    elif args.dataset == "bigbench_date":
        pred = re.findall(r'A|B|C|D|E|F', pred)
    elif args.dataset in ("object_tracking",):
        pred = re.findall(r'A|B|C', pred)
    elif args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
        if must_choice:
            pred = re.findall(r'A|B|C|D', pred)
        else:
            if not answer_flag:
                sentences = re.split(r'[.!?]', pred)
                last_two = ' '.join(sentences[-2:]) if len(sentences) >= 2 else pred
                pred = re.findall(r'-?\d+(?:\.\d+)?', last_two.replace(',', ''))
            else:
                pred = re.findall(r'-?\d+(?:\.\d+)?', pred.replace(',', ''))
    elif args.dataset in ("strategyqa", "coin_flip"):
        pred = pred.lower()
        pred = re.sub(r'["\'\n.\s:!,]', ' ', pred)
        pred = [tok for tok in pred.split() if tok in ("yes", "no")]
    elif args.dataset in ("last_letters",):
        pred = re.sub(r'["\'\n.\s]', '', pred)
        pred = [pred]
    else:
        raise ValueError("dataset is not properly defined ...")

    if len(pred) == 0:
        answer_cleansing.fail_count += 1
        pred = ""
    else:
        if args.method in ("few_shot", "few_shot_cot", "auto_cot", "pattern_cot", "test_cot"):
            pred = pred[0] if answer_flag else pred[-1]
        else:                   # zero-shot 类
            pred = pred[0]

    if pred and pred[-1] == '.':
        pred = pred[:-1]

    print("pred_after  : " + pred)
    if pred == "":
        print("[AnswerCleansing] failed to extract any valid answer.")
    return pred


def get_answer_cleansing_fail_count():
    return getattr(answer_cleansing, 'fail_count', 0)


def create_demo_text(args, cot_flag):
    x, z, y = [], [], []
    
    with open(args.demo_path, encoding="utf-8") as f:
        json_data = json.load(f)
        json_data = json_data["demo"]
        for line in json_data:
            x.append(line["question"])
            z.append(line["rationale"])
            y.append(line["pred_ans"])

    index_list = list(range(len(x)))
    
    demo_text = ""
    for i in index_list:
        if cot_flag:
            demo_text += x[i] + " " + z[i] + " " + \
                         args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            demo_text += x[i] + " " + args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
    return demo_text

def answer_cleansing_zero_shot(args, pred, must_choice=False):
    pred = pred.strip()
    if args.dataset in ("aqua", "commonsensqa"):
        pred = re.findall(r'A|B|C|D|E', pred)
    elif args.dataset == "bigbench_date":
        pred = re.findall(r'A|B|C|D|E|F', pred)
    elif args.dataset in ("object_tracking"):
        pred = re.findall(r'A|B|C', pred)
    elif args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
        if must_choice:
            pred = re.findall(r'A|B|C|D', pred)
        else:
            pred = pred.replace(",", "")
            pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif args.dataset in ("strategyqa", "coin_flip"):
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]
    elif args.dataset in ("last_letters"):
        pred = re.sub("\"|\'|\n|\.|\s", "", pred)
        pred = [pred]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        # choose the first element in list ...
        pred = pred[0]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]

    return pred


def convert_gsm8k_jsonl_to_json(input_file: str, output_file: str):
    # INPUT_FILE  = Path('experiment/gsm8k_zero_shot_llama13b.jsonl')
    # OUTPUT_FILE = Path('gsm8k_llama13b.json')

    INPUT_FILE = Path(input_file)
    OUTPUT_FILE = Path(output_file)

    demo = []
    with INPUT_FILE.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            demo.append(
                {
                    "question":  obj.get("question", ""),
                    "rationale": obj.get("rationale", ""),
                    "pred_ans":  obj.get("pred_ans", ""),
                    "gold_ans":  obj.get("gold_ans", ""),
                    "wrap_que":  obj.get("wrap_que", "")
                }
            )

    with OUTPUT_FILE.open('w', encoding='utf-8') as f_out:
        json.dump({"demo": demo}, f_out, ensure_ascii=False, indent=2)
    return len(demo)