## Evaluation


```bash
python eval.py
```



## Vllm Inference

You can set `use_api = False` to evaluate with vllm inference. 

Set basic configurations such like model name `pretrained_model` and max token the model can generate for each quiry `max_generation_token`.

Set your GPU setting `gpu_groups`. For example, [[0]] represents only one engine with one GPU; [[0, 1], [2, 3], [4, 5], [6, 7]] represents four engines each with 2 GPUs. Each engine loads a model, so for <=7B model, you can only set GPU numbers for each engine <= 2.


## API Inference


You can also do inference with API. Set `api_key`, `base_url` and the model name `api_model_name`.

By default, we set `use_openai_batch_api = False`, and control the request per minute limit by `rpm_limit`. However, if it's OpenAI's model, and your account is available for batch inference, recommend setting `use_openai_batch_api = True`, it's cheaper and faster.


























