# ========================================= config for optimization process ==========================================
# ====================================================================================================================


# the model you want to optimize
pretrained_model = "Qwen/Qwen2.5-7B-Instruct"


# the training data and evaluation data
train_dataset = "MATH_train"
eval_dataset = "MATH500"

# total steps for optimization
total_steps = 120

# evaluate every eval_interval steps
eval_interval = 10

# save optimized model every save_interval steps
save_interval = 40









# ============= config for sampling in each step =================

# number of samples
k_sample = 16

# temperature
temp = 1.0

# number of tasks for sampling in each step
n_sample_per_step = 100

# GPU usage for vllm inference, [[0]] represents only one engine with one GPU; [[0, 1], [2, 3]] represents two engines each with 2 GPUs
# each engine loads a model, so for <=7B model, you can only set GPU numbers for each engine <= 2
gpu_groups = [[0,1],[2,3],[4,5],[6,7]]

# maximum number of tokens the vLLM engine can handle in a single sequence
max_model_len = 10000

# max token model can generate for each quiry
max_generation_token = 2500

# set False by default
if_start_with_think = False

# the prompt design for code generation and unit test generation
system_prompts = """<|im_start|>You are a helpful assistant help user solve problems. <|im_end|>\n<|im_start|>user 
You need to put your final answer in \\boxed{}. This is the problem:\n{{problem}} <|im_end|>\n<|im_start|>assistant 
"""
























# ============= config for reward assignment in each step =================



















# ============= config for training in each step =================

# number of GPUs
total_num_nodes = 8

# learning rate
actor_learning_rate = 1e-6

# 0 by default
num_warmup_steps = 0

# number of updates each step, 1 by default
policy_update_steps = 1

# KL loss setting
use_kl_loss = True
kl_loss_coef = 0.01
use_kl_estimator_k3 = True

# max prompt (inquiry) length in collected data
prompt_max_len = 2000

# generation token limit
generate_max_len = 2500

# we use packing here instead of batching for training, and we need packing_max_len >= generate_max_len + prompt_max_len
packing_max_len = 20000

# number of epoch for this training, 1 by default
max_epochs = 1

# the output model name
optimized_model_name = "optimized"










