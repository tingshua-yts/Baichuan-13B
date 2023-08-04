from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import torch
import os
def fullname(o):
    klass = o.__class__
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__ # avoid outputs like 'builtins.str'
    print(os.path.abspath(module.__file__))
    return module + '.' + klass.__qualname__

model_name = "/mnt/model/Baichuan-13B-Chat/"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(model_name)
messages = []
messages.append({"role": "user", "content": "世界上第二高的山峰是哪座"})
print(model)
print(fullname(model))

response = model.chat(tokenizer, messages)
print(response)