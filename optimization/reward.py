import os
import ast
import json
import random
import asyncio
import argparse
import math_utils
import numpy as np
import nest_asyncio
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor


import optimization_config





# read the configurations and load them as global variables

def str2bool(x):
    return x.lower() in ("1", "true", "yes")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default=optimization_config.pretrained_model)
    parser.add_argument("--dataset", type=str, default=optimization_config.train_dataset)
    parser.add_argument("--max_generation_len", type=int, default=optimization_config.max_generation_token)
    return parser.parse_args()

args = parse_args()
globals().update(vars(args))



# read the inference data

outputs_name =  pretrained_model.replace("/", ".") + "-" + dataset

os.makedirs(os.path.dirname("./temp_data/outputs-rl-" + outputs_name + ".json"), exist_ok=True)
with open("./temp_data/outputs-rl-" + outputs_name + ".json", 'r') as f:
    data = json.load(f)
    
os.makedirs(os.path.dirname("../temp_data/rl_data_response.json"), exist_ok=True)
with open("./temp_data/rl_data_response.json", 'r') as f:
    data_pit = json.load(f)

nest_asyncio.apply()

def flatten_correctness_inputs(data_list):
    index_list = []
    extracted_output_list = []
    ground_truth_list = []

    for i, item in enumerate(data_list):
        item["correctness"] = []
        num_outputs = len(item["extracted_output"])
        index_list.extend([i] * num_outputs)
        extracted_output_list.extend(item["extracted_output"])
        ground_truth_list.extend([item["ground_truth_answer"]] * num_outputs)
    
    return index_list, extracted_output_list, ground_truth_list

async def compute_correctness_async(extracted_outputs, ground_truths):
    executor = ThreadPoolExecutor(max_workers=64)
    tasks = [
        math_utils.is_equal(pred, gt, executor)
        for pred, gt in zip(extracted_outputs, ground_truths)
    ]
    return await asyncio.gather(*tasks)

def compute_correctness(data_list):
    index_list, extracted_outputs, ground_truths = flatten_correctness_inputs(data_list)
    correctness_list = asyncio.run(compute_correctness_async(extracted_outputs, ground_truths))
    for idx, corr in zip(index_list, correctness_list):
        data_list[idx]["correctness"].append(corr)


compute_correctness(data)
compute_correctness(data_pit)


def z_score_normalize(lst):
    mean = sum(lst) / len(lst)
    std = (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5
    if std == 0:
        return [0 for x in lst]
    return [(x - mean) / std for x in lst]

all_data = []
pit_data = []
final_data = []

# reward for all responses (n + m * k)
for i in range(len(data)):
    correctness = data[i]["correctness"] 
    lengths = data[i]["response_length"]
    rewards = z_score_normalize(correctness)
    data[i]["rewards"] = rewards
    
    
    for j in range(len(rewards)): 
        data_i = {}
        data_i["prompt"] = data[i]["prompt"] 
        
        data_i["reward"] = rewards[j]
        if lengths[j] < max_generation_len:
            data_i["response"] = data[i]["full_output"][j] + "<|im_end|>" 
        else:
            data_i["response"] = data[i]["full_output"][j] 
        all_data.append(data_i)

# reward for pits
for i in range(len(data_pit)):
    correctness = data_pit[i]["correctness"] 
    lengths = data_pit[i]["response_length"]
    reward = sum(correctness) / len(correctness)
    data_pit[i]["rewards"] = reward
    
    data_i = {}
    data_i["prompt"] = data_pit[i]["prompt"] 
    data_i["response"] = data_pit[i]["response"] 
    data_i["reward"] = reward
        
    pit_data.append(data_i)
    
final_data = all_data + pit_data

with open("./temp_data/rl_data.json", "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

os.makedirs(os.path.dirname("./temp_data/outputs-" + outputs_name + ".json"), exist_ok=True)
with open("./temp_data/outputs-rl-" + outputs_name + ".json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
