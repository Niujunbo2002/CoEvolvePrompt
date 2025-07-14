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


index_list = []
extracted_output_list = []
ground_truth_list = []
for i in range(len(data)):
    data[i]["correctness"] = []
    index_list = index_list + [i] * len(data[i]["extracted_output"])
    extracted_output_list = extracted_output_list + data[i]["extracted_output"]
    ground_truth_list = ground_truth_list + [data[i]["ground_truth_answer"]] * len(data[i]["extracted_output"])

nest_asyncio.apply()

async def get_correctness():
    executor = ThreadPoolExecutor(max_workers=64)
    tasks = []
    for i in range(len(index_list)):
        tasks.append(math_utils.is_equal(extracted_output_list[i], ground_truth_list[i], executor))
    results = await asyncio.gather(*tasks)
    return results

correctness_list = asyncio.run(get_correctness())
for i in range(len(index_list)):
    index_i = index_list[i]
    data[index_i]["correctness"].append(correctness_list[i])



def z_score_normalize(lst):
    mean = sum(lst) / len(lst)
    std = (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5
    if std == 0:
        return [0 for x in lst]
    return [(x - mean) / std for x in lst]

final_data = []
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
        final_data.append(data_i)


with open("./temp_data/rl_data.json", "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

os.makedirs(os.path.dirname("./temp_data/outputs-" + outputs_name + ".json"), exist_ok=True)
with open("./temp_data/outputs-rl-" + outputs_name + ".json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
