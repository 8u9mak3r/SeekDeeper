import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer, AutoModelForMaskedLM, TorchAoConfig
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, Int8WeightOnlyConfig
import torch

model_id = "bert-base-uncased"

quant_config = Int8WeightOnlyConfig()
quantization_config = TorchAoConfig(quant_type=quant_config)

# 1. 配置量化策略
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载 tokenizer 和量化模型
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    # device_map="auto"
).eval()

text = "The capital of France is [MASK]."
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 取出 mask token 的预测结果
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = logits[0, mask_token_index].argmax(dim=-1)
predicted_word = tokenizer.decode(predicted_token_id)

print(f"Predicted: {predicted_word}")
