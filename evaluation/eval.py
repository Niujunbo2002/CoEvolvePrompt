import os
import re
import io
import sys
import ast
import json
import time
import random
import typing
import asyncio
import argparse
import requests
import math_utils
import numpy as np
import nest_asyncio
from openai import OpenAI
from jinja2 import Template
from termcolor import cprint
import multiprocessing as mp
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed

import evaluation_config




os.environ["TOKENIZERS_PARALLELISM"] = "false" 







#============== vllm inference ===============

def worker_fn(pretrained_model, gpu_ids, task_queue, result_queue, max_model_len, max_generation_token):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    print(f"Loading model on GPUs {gpu_ids}...")
    kwargs = {}
    llm = LLM(
        model=pretrained_model,
        dtype="bfloat16",
        tensor_parallel_size=len(gpu_ids),
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len,
        **kwargs
    )

    sampling_params = SamplingParams(
        temperature=temp,
        top_p=0.95,
        top_k=40,
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

# get token length
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















#============== API inference ===============

def fetch_completion(user_prompt: str) -> str:

    headers  = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": api_model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user",   "content": user_prompt}
        ],
        "temperature": api_temperature
    }
    r = requests.post(base_url, headers=headers, json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]



def generate_results_api(prompts):
    total   = len(prompts)
    results = ["No outputs"] * total

    for batch_start in range(0, total, rpm_limit):
        batch_end   = min(batch_start + rpm_limit, total)
        batch_slice = range(batch_start, batch_end)

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            fut_to_idx = {
                pool.submit(fetch_completion, prompts[i]): i
                for i in batch_slice
            }

            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                results[idx] = fut.result()

        elapsed   = time.time() - t0
        leftover  = max(0, 60.0 - elapsed)
        if batch_end < total and leftover:
            time.sleep(leftover)

        print(f"Processed {batch_end}/{total} prompts")

    return results


def save_prompts_to_jsonl(prompts,
                         filename,
                         system_content,
                         model,
                         max_tokens,
                         url):
    with open(filename, "w", encoding="utf-8") as fout:
        for i, user_prompt in enumerate(prompts, start=1):
            obj = {
                "custom_id": f"request-{i}",
                "method":    "POST",
                "url":       url,
                "body": {
                    "model":      model,
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user",   "content": user_prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": api_temperature
                }
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote {len(prompts)} requests to {filename!r}")

def extract_completions(raw):
    lines = raw.strip().split("\n")
    records = [json.loads(line) for line in lines]
    bodies = [rec["response"]["body"] for rec in records]
    assistant_texts = [
        body["choices"][0]["message"]["content"]
        for body in bodies
    ]
    return assistant_texts

def generate_by_openai_batch(prompts):

    save_prompts_to_jsonl(
        prompts,
        filename=api_batch_filename,
        system_content="You are a helpful assistant.",
        model=api_model_name,
        max_tokens=max_tokens,
        url="/v1/chat/completions"
    )

    client = OpenAI(
        api_key=api_key,
    )

    batch_input_file = client.files.create(
        file=open(api_batch_filename, "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id
    cprint(f"file id: {batch_input_file_id}", color = "green")
    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
        }
    )
    batch_id = batch.id
    cprint(f"batch id: {batch_id}", color = "green")
    cprint("You can check https://platform.openai.com/docs/guides/batch?lang=python to learn how to monitor and cancel this batch job with batch id and file id.", color = "green")

    import time
    start_time = time.time()
    last_index = 0
    min_interval = 2
    while True:
        time.sleep(5)
        batch = client.batches.retrieve(batch_id)
        if batch.status == "completed":
            file_id = batch.output_file_id
            break
        if batch.status == "failed" or batch.status == "expired" or batch.status == "cancelled":
            cprint(batch.status, color = "green")
            return None
        elapsed = time.time() - start_time
        idx = int(elapsed // (60 * min_interval))
        if idx > last_index:
            last_index = idx
            num_completed = batch.request_counts.completed
            total_num = batch.request_counts.total
            failed_num = batch.request_counts.failed
            print(f"{idx * min_interval} minutes passed, {num_completed}/{total_num} completed, {failed_num} failed, {batch.status}")

    cprint(f"takes {time.time() - start_time}s to complete!", color = "green")
    file_response = client.files.content(file_id)
    return extract_completions(file_response.text)




# read the configuration
def str2bool(x):
    return x.lower() in ("1", "true", "yes")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default=evaluation_config.pretrained_model)
    parser.add_argument("--dataset", type=str, default=evaluation_config.dataset)
    parser.add_argument("--use_api", type=str2bool, default=evaluation_config.use_api)
    parser.add_argument("--if_start_with_think", type=str2bool, default=evaluation_config.if_start_with_think)
    parser.add_argument("--max_model_len", type=int, default=evaluation_config.max_model_len)
    parser.add_argument("--max_generation_token", type=int, default=evaluation_config.max_generation_token)
    parser.add_argument("--temp", type=float, default=evaluation_config.temp)
    parser.add_argument("--is_final_eval", type=str2bool, default=evaluation_config.is_final_eval)
    parser.add_argument("--api_model_name", type=str, default=evaluation_config.api_model_name)
    parser.add_argument("--api_key", type=str, default=evaluation_config.api_key)
    parser.add_argument("--base_url", type=str, default=evaluation_config.base_url)
    parser.add_argument("--api_temperature", type=float, default=evaluation_config.api_temperature)
    parser.add_argument("--max_workers", type=int, default=evaluation_config.max_workers)
    parser.add_argument("--use_openai_batch_api", type=str2bool, default=evaluation_config.use_openai_batch_api)
    parser.add_argument("--max_tokens", type=int, default=evaluation_config.max_tokens)
    parser.add_argument("--rpm_limit", type=int, default=evaluation_config.rpm_limit)
    parser.add_argument("--gpu_groups", type=ast.literal_eval, default=evaluation_config.gpu_groups)
    parser.add_argument("--system_prompts", type=str, default=evaluation_config.system_prompts)
    return parser.parse_args()


# convert read configuration to global variable
args = parse_args()
globals().update(vars(args))





# read dataset
with open("../data/" + dataset + ".json", 'r') as f:
    data = json.load(f)
#data = [data[i] for i in range(10)]
num = len(data)


# load model, tokenizer, build vllm engines...
if use_api == False:
    task_queues, result_queues, processes = start_workers(pretrained_model, gpu_groups, max_model_len, max_generation_token)
    outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
else:
    outputs_name = "eval-" + api_model_name.replace("/", ".") + "-" + dataset
    if use_openai_batch_api:
        api_batch_filename = api_model_name.replace("/", ".") + "-" + dataset + ".jsonl"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


if if_start_with_think:
    system_prompts = system_prompts + "<think>\n"
    system_case_prompts = system_case_prompts + "<think>\n"


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
    generation_prompts = generation_prompts + [get_prompt(data[i])]
    index_list = index_list + [i]
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









# calculate response length

def get_token_lengths(strings, tokenizer):
    return [len(tokenizer.encode(s, add_special_tokens=False)) for s in strings]

response_length = get_token_lengths(restored_outputs, tokenizer)
mean_response_length = sum(response_length)/len(response_length)



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


if use_api == False:
    stop_workers(task_queues, processes)






# derive results

with open("./temp_data/outputs-" + outputs_name + '.json', 'r') as f:
    data = json.load(f)


index_list = []
extracted_output_list = []
ground_truth_list = []
response_length_list = []
for i in range(len(data)):
    data[i]["correctness"] = []
    index_list = index_list + [i] * len(data[i]["extracted_output"])
    extracted_output_list = extracted_output_list + data[i]["extracted_output"]
    ground_truth_list = ground_truth_list + [data[i]["ground_truth_answer"]] * len(data[i]["extracted_output"])
    response_length_list = response_length_list + data[i]["response_length"]

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


if is_final_eval:
    if use_api == False:
        outputs_result_name = "./results/results-eval-" + pretrained_model.replace("/", ".") + "-final_eval.txt"
    else:
        outputs_result_name = "./results/results-eval-" + api_model_name.replace("/", ".") + "-final_eval.txt"
else:
    outputs_result_name = "./results/results-" + outputs_name + ".txt"
os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)
with open(outputs_result_name, "a") as f:
    # Save + print
    def save_and_print(text):
        cprint(text, color="green")
        f.write(text + "\n")

    # Your values
    def safe_divide(d1, d2):
        if d2 == 0:
            return 0
        return d1/d2
    
    save_and_print(f"acc: {sum(correctness_list) / len(correctness_list)}")
    save_and_print(f"average response length: {sum(response_length_list) / len(response_length_list)}")










