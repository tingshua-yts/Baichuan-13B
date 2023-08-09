from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from perf_run import measure_python_inference_code, TimingProfile
tokenizer = AutoTokenizer.from_pretrained("/mnt/model/Baichuan-13B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/mnt/model/Baichuan-13B-Chat", torch_dtype=torch.float16,device_map="auto", trust_remote_code=True)

inputs = tokenizer('帮我写一个简短的程序员笑话', return_tensors='pt')
input_ids_len = len(inputs["input_ids"][0])
inputs = inputs.to('cuda:0')

pred = model.generate(**inputs, max_new_tokens=128,repetition_penalty=1.1)

print(f"input_ids len: {input_ids_len}, output_ids len: {len(pred[0])}")


perf_model = lambda: model.generate(**inputs, max_new_tokens=128,repetition_penalty=1.1)

perf_result = measure_python_inference_code(perf_model, TimingProfile(iterations=10, number=1, warmup=1, duration=0, percentile=[50, 95]))
print(perf_result)