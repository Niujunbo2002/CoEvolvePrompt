# ========================================= config for optimization process ==========================================
# ====================================================================================================================


# the model you want to optimize
pretrained_model = "/data1/niujunbo/models/Qwen/Qwen2.5-0.5B-Instruct" # Qwen2.5-0.5B-Instruct; qwen-2.5-7b-instruct


# the training data and evaluation data
train_dataset = "MATH_train"
eval_dataset = "MATH500"

# total steps for optimization
total_steps = 120#120

# evaluate every eval_interval steps
eval_interval = 5#10

# save optimized model every save_interval steps
save_interval = 10#40









# ============= config for sampling in each step =================

# number of samples
k_sample = 16

# number of re-generated answers for each wrong answer
m_regenerate = 4

# temperature
temp = 1.0

# number of tasks for sampling in each step
n_sample_per_step = 100

# GPU usage for vllm inference, [[0]] represents only one engine with one GPU; [[0, 1], [2, 3]] represents two engines each with 2 GPUs
# each engine loads a model, so for <=7B model, you can only set GPU numbers for each engine <= 2
gpu_groups = [[4,5],[2,3]]

# maximum number of tokens the vLLM engine can handle in a single sequence
max_model_len = 3000

# max token model can generate for each quiry
max_generation_token = 1500

# set False by default
if_start_with_think = False

# the prompt design for code generation and unit test generation
system_prompts = """<|im_start|>You are a helpful assistant help user solve problems. <|im_end|>\n<|im_start|>user 
You need to put your final answer in \\boxed{}. This is the problem:\n{{problem}} <|im_end|>\n<|im_start|>assistant 
"""


pit_prompts = """
<|im_start|>system
You are a self-reflective assistant. You previously answered a question incorrectly. 
First, analyze the root cause of your mistake (e.g., calculation error, logic flaw, misunderstanding of the problem).
Then, based on that analysis, generate a new prompt to guide future responses.

IMPORTANT:
- The new prompt must be wrapped in \\boxed_prompt{...}
- Do NOT include the correct answer in the analysis or the new prompt.
- Your final output should include:
  (1) A concise analysis of the mistake
  (2) The new prompt in \\boxed_prompt{...}

Here is an example of the expected format:

[Analysis]
I mistakenly assumed that the triangle was a right triangle without checking. This led me to incorrectly apply the Pythagorean theorem.

[New Prompt]
\\boxed_prompt{Always verify the type of triangle before applying specific formulas.}
<|im_end|>

<|im_start|>user
Problem: \n{{problem}}\n
Your incorrect answer: \n{{incorrect_answer}}\n
Correct answer: \n{{correct_answer}} (for analysis only — do not mention this directly)\n

Now analyze the mistake and generate a new prompt:
<|im_end|>
\n<|im_start|>assistant
"""



#Your full incorrect response: \n{{full_incorrect_response}}

























# ============= config for reward assignment in each step =================



















# ============= config for training in each step =================

# number of GPUs
total_num_nodes = 4

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
packing_max_len = 5000#20000

# number of epoch for this training, 1 by default
max_epochs = 1

# the output model name
optimized_model_name = "Co_optimized_debug"










