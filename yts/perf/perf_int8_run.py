from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import torch
from perf_run import measure_python_inference_code, TimingProfile
model_name = "/mnt/model/Baichuan-13B-Chat/"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
model = model.quantize(8).cuda()
model.generation_config = GenerationConfig.from_pretrained(model_name)
messages = []
messages.append({"role": "user", "content": "世界上第二高的山峰是哪座"})
response = model.chat(tokenizer, messages)
print(response)

print(f"model device: {model.device}")
inputs = tokenizer('帮我写一个简短的程序员笑话', return_tensors='pt')
inputs = inputs.to('cuda:0')
pred = model.generate(**inputs, max_new_tokens=64,repetition_penalty=1.1)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))

### perf
perf_model = lambda: model.generate(**inputs, max_new_tokens=128,repetition_penalty=1.1)

perf_result = measure_python_inference_code(perf_model, TimingProfile(iterations=10, number=1, warmup=1, duration=0, percentile=[50, 95]))
### print result
print(perf_result)