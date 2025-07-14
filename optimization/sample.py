import os
import re
import ast
import json
import random
import argparse
from jinja2 import Template
from termcolor import cprint
import multiprocessing as mp
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import optimization_config





os.environ["TOKENIZERS_PARALLELISM"] = "false" 





####### vllm inference #######

def worker_fn(pretrained_model, gpu_ids, task_queue, result_queue, max_model_len, max_generation_token):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    print(f"Loading model on GPUs {gpu_ids}...")
    llm = LLM(
        model=pretrained_model,
        dtype="bfloat16",
        tensor_parallel_size=len(gpu_ids),
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len
    )

    sampling_params = SamplingParams(
        temperature=temp,
        top_p=0.95,
        top_k=-1,
        min_p=0.0,
        max_tokens=max_generation_token,
        stop=["</answer>", "User:", "Human:", "Assistant:", "<|im_end|>", "<|endoftext|>"]
    )

    while True:
        task = task_queue.get()
        if task == "STOP":
            print("Stopping worker...")
            break
        task_id, prompts = task
        outputs = llm.generate(prompts, sampling_params)
        result_texts = [out.outputs[0].text for out in outputs]
        result_queue.put((task_id, result_texts))

# To run the worker setup:
def start_workers(pretrained_model, gpu_configs, max_model_len, max_generation_token):
    task_queues = []
    result_queues = []
    processes = []

    for i, gpu_ids in enumerate(gpu_configs):
        task_q = mp.Queue()
        result_q = mp.Queue()
        p = mp.Process(
            target=worker_fn,
            args=(pretrained_model, gpu_ids, task_q, result_q, max_model_len, max_generation_token)
        )
        p.start()
        task_queues.append(task_q)
        result_queues.append(result_q)
        processes.append(p)
    
    return task_queues, result_queues, processes

# Submit tasks
def submit_prompt_set(task_queues, prompt_sets):
    for i, prompts in enumerate(prompt_sets):
        task_queues[i].put((i, prompts))

# Collect results
def collect_results(result_queues, num_sets):
    results = [None] * num_sets
    for q in result_queues:
        task_id, result = q.get()
        results[task_id] = result
    return results

# Stop workers
def stop_workers(task_queues, processes):
    for q in task_queues:
        q.put("STOP")
    for p in processes:
        p.join()

# Split prompts into N chunks
def split_prompts(prompts, n):
    k, m = divmod(len(prompts), n)
    return [prompts[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def get_token_lengths(strings, tokenizer):
    return [len(tokenizer.encode(s, add_special_tokens=False)) for s in strings]

# vllm inference
def generate_results(all_prompts, gpu_groups,task_queues, result_queues):
    prompt_sets = split_prompts(all_prompts, len(gpu_groups))
    submit_prompt_set(task_queues, prompt_sets)
    results = collect_results(result_queues, len(prompt_sets))
    result_list = []
    for result_set in results:
        for r in result_set:
            result_list.append(r)
    return result_list


import random 
def random_select(data_list, random_k):
    data_list = random.sample(data_list, random_k)
    return data_list




# read the condiguration and convert them into global variables

def str2bool(x):
    return x.lower() in ("1", "true", "yes")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default=optimization_config.pretrained_model)
    parser.add_argument("--dataset", type=str, default=optimization_config.train_dataset)
    parser.add_argument("--if_start_with_think", type=str2bool, default=optimization_config.if_start_with_think)
    parser.add_argument("--max_model_len", type=int, default=optimization_config.max_model_len)
    parser.add_argument("--max_generation_token", type=int, default=optimization_config.max_generation_token)
    parser.add_argument("--temp", type=float, default=optimization_config.temp)
    parser.add_argument("--k_sample", type=int, default=optimization_config.k_sample)
    parser.add_argument("--random_select_num", type=int, default=optimization_config.n_sample_per_step)
    parser.add_argument("--gpu_groups", type=ast.literal_eval, default=optimization_config.gpu_groups)
    parser.add_argument("--system_prompts", type=str, default=optimization_config.system_prompts)
    return parser.parse_args()

args = parse_args()
globals().update(vars(args))










# read dataset
with open("../data/" + dataset + ".json", 'r') as f:
    data = json.load(f)
#data = [data[i] for i in range(10)]
random_select_num = min(random_select_num, len(data))
data = random_select(data, random_select_num)
num = len(data)


# load model, tokenizer, build vllm engines...
task_queues, result_queues, processes = start_workers(pretrained_model, gpu_groups, max_model_len, max_generation_token)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset


if if_start_with_think:
    system_prompts = system_prompts + "<think>\n"
    system_case_prompts = system_case_prompts + "<think>\n"







def bernoulli(p):
    return 1 if random.random() < p else 0

# obtain prompt
def get_prompt(data_i):
    return Template(system_prompts).render(problem = data_i["question"])


def extract_final_boxed_answer(s: str):
    tag = r'\boxed{'
    start = s.rfind(tag)          # last \boxed{
    if start == -1:
        return "Can not extract the answer!"

    i = start + len(tag)
    depth = 1                    # we are already inside one '{'
    buf = []

    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:       # matching '}' for the opening \boxed{
                break
        buf.append(ch)
        i += 1

    return ''.join(buf) if depth == 0 else "Can not extract the answer!"





# initialization
generation_prompts = []
index_list = []
for i in range(num):
    # preprocess
    generation_prompts = generation_prompts + [get_prompt(data[i])] * k_sample
    index_list = index_list + [i] * k_sample
    data[i]["full_output"] = []
    data[i]["extracted_output"] = []
    data[i]["response_length"] = []
    data[i]["prompt"] = get_prompt(data[i])








# sampling process

cprint("start generation...", "green")

# shuffle first, to achieve efficiency
all_prompts = generation_prompts
N = len(all_prompts)
indices = list(range(N))
shuffled_idx = indices[:]      
random.shuffle(shuffled_idx)
shuffled_prompts = [all_prompts[i] for i in shuffled_idx]
# generate
shuffled_outputs = generate_results(shuffled_prompts, gpu_groups, task_queues, result_queues)
restored_outputs = [None] * N
for out, idx in zip(shuffled_outputs, shuffled_idx):
    restored_outputs[idx] = out

cprint("generation job done!", "green")










# calculate the response length

def get_token_lengths(strings, tokenizer):
    return [len(tokenizer.encode(s, add_special_tokens=False)) for s in strings]

response_length = get_token_lengths(restored_outputs, tokenizer)
mean_response_length = sum(response_length)/len(response_length)

os.makedirs(os.path.dirname("./results/results-" + outputs_name + ".txt"), exist_ok=True)
with open("./results/results-" + outputs_name + ".txt", "a") as f:
    # Save + print
    def save_and_print(text):
        cprint(text, color="green")
        f.write(text + "\n")
    save_and_print(f"response length: {mean_response_length}")





# process generated codes
i = 0
for full_output in restored_outputs:
    extracted_output = extract_final_boxed_answer(full_output)
    index_i = index_list[i]
    data[index_i]["full_output"].append(full_output)
    data[index_i]["extracted_output"].append(extracted_output)
    data[index_i]["response_length"].append(response_length[i])
    i += 1


# output the data
os.makedirs(os.path.dirname("./temp_data/outputs-" + outputs_name + ".json"), exist_ok=True)
with open("./temp_data/outputs-" + outputs_name + ".json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)



stop_workers(task_queues, processes)










