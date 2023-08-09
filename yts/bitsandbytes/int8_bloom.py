from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import torch
import sys
import time
sys.path.append("/mnt/project/Baichuan-13B/yts")
from perf.perf_run import measure_python_inference_code, TimingProfile
model_name = "/mnt/model/bloom-7b1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True,trust_remote_code=True)
inputs = tokenizer('编写一个程序员笑话', return_tensors='pt')
inputs = inputs.to('cuda:0')
pred = model.generate(**inputs, max_new_tokens=64,repetition_penalty=1.1)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))

##### perf
### perf
perf_model = lambda: model.generate(**inputs, max_new_tokens=128,repetition_penalty=1.1)

perf_result = measure_python_inference_code(perf_model, TimingProfile(iterations=10, number=1, warmup=1, duration=0, percentile=[50, 95]))
### print result
print(perf_result)
time.sleep(3600)