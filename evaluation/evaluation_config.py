# ========================  This is the config for evaluation  =========================
# ======================================================================================


# set True if you want to evaluate API model, set False for vllm inference model
use_api = False


# dataset you can directly use for evaluation: CodeContests_test, LiveBench, LiveCodeBench, Codeforces, MBPP
dataset = "MATH500"

# set True by default here, no need to change
is_final_eval = True
















# ======================== config for vllm inference model (use_api = False) ========================


# vllm model name
#pretrained_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
pretrained_model = "../optimization/ckpt/optimized"

# maximum number of tokens the vLLM engine can handle in a single sequence
max_model_len = 20000

# max token model can generate for each quiry
max_generation_token = 10000

# inference temperature
temp = 0.8

# set False by default
if_start_with_think = False

# GPU usage for vllm inference, [[0]] represents only one engine with one GPU; [[0, 1], [2, 3]] represents two engines each with 2 GPUs
# each engine loads a model, so for <=7B model, you can only set GPU numbers for each engine <= 2
gpu_groups = [[0,1],[2,3],[4,5],[6,7]]















# ======================== config for API inference model (use_api = True) ========================


# api_key and base_url
api_key = "Your API Key"
base_url = "Base URL, For Example, https://api.openai.com/v1/chat/completions"

# api model name, such like "gpt-4o", "deepseek-chat"
api_model_name = "gpt-4o-mini"

# temperature
api_temperature = 0.8

# max inquiries submitted at one time
max_workers = 20

# if it's OpenAI's model, and your account is available for batch inference, recommend setting this to be True, it's cheaper
use_openai_batch_api = False

# max token can generate for each task
max_tokens = 2500

# the request per minute limit for your API
rpm_limit = 100













# ======================= the prompt for code generation and unit test generation ============================

# the prompt design for code generation and unit test generation
system_prompts = """<|im_start|>You are a helpful assistant help user solve problems. <|im_end|>\n<|im_start|>user 
You need to put your final answer in \\boxed{}. This is the problem:\n{{problem}} <|im_end|>\n<|im_start|>assistant 
"""







