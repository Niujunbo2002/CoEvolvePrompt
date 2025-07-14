# Optimization



## Main Configurations

`pretrained_model`: name of the model to be trained

`train_dataset` and `eval_dataset`: training and evaluation dataset

`eval_interval`: evaluate every eval_interval steps

`save_interval`: save optimized model every save_interval steps

## Sampling

Module: `sample.py`

`k_sample` : number of sampled solutions in each step

`n_sample_per_step`: number of tasks for sampling in each step

`gpu_groups`: GPU usage for vllm inference. For example, [[0]] represents only one engine with one GPU; [[0, 1], [2, 3]] represents two engines each with 2 GPUs




## Reward Assignment

Module: `reward.py`

Set `rank_reward = True` to do ranking reward, otherwise length-penalty.

## Train

Module: `train.py`

`total_num_nodes`: number of GPUs

`actor_learning_rate`: learning rate

`prompt_max_len`: max prompt (inquiry) length in collected data

`generate_max_len`: generation token limit

`packing_max_len`: we use packing here instead of batching for training, and we need packing_max_len >= generate_max_len + prompt_max_len

`optimized_model_name`: the output model name, the model will be saved under `./ckpt`







