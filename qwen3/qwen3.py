import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

import transformers
print(transformers.__version__)  # 必须 >= 4.51.0


# ————————————————————————————————
# 1. 加载模型与分词器
# ————————————————————————————————
model_name = "Qwen/Qwen3-0.6B"
tokenizer  = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model      = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ————————————————————————————————
# 2. 定义 intermediates 字典和 Hook 函数
# ————————————————————————————————
intermediates = {}
pre_inputs = {}

def save_tensor_hook(module, inputs, output):
    if isinstance(output, torch.Tensor):
        intermediates[module] = output.detach().cpu().numpy()
    else:
        # 如果 output 是 tuple，可以根据需要调整
        intermediates[module] = output[0].detach().cpu().numpy()

def pre_hook(module, inputs):
    tensor = None
    for item in inputs:
        if isinstance(item, torch.Tensor):
            tensor = item
            break
    if tensor is None:
        return
    arr = tensor.detach().cpu().float().numpy()
    pre_inputs[module] = arr
# ————————————————————————————————
# 3. 注册 Hook（拿到所有 self_attn 子模块的输出）
# ————————————————————————————————
for name, module in model.named_modules():
    module.register_forward_hook(save_tensor_hook)
    module.register_forward_pre_hook(pre_hook)
    print(f"[Hook 已注册] -> {name}")

# ————————————————————————————————
# 4. 准备输入并跑一次推理
# ————————————————————————————————
text = "this part"
inputs = tokenizer(text, return_tensors="pt").to(device)
print(inputs)

with torch.no_grad():
    _ = model(**inputs)

# ————————————————————————————————
# 5. 打印 intermediates 里存储的张量“数值示例”
# ————————————————————————————————
for module, arr in intermediates.items():
    print("─" * 60)
    print(f"模块: {module}")
    print(f"张量形状: {arr.shape}")
    if module in pre_inputs:
        print("运算前：")
        flat = pre_inputs[module][0, 1].flatten()
        print("前 10 个元素(展平后)：", flat[:10].tolist())
        print("后 10 个元素(展平后)：", flat[-10:].tolist())
        print('运算后:')
    flat = arr[0, 1].flatten()
    print("前 10 个元素(展平后)：", flat[:10].tolist())
    print("后 10 个元素(展平后)：", flat[-10:].tolist())
    arr = pre_inputs.get(module)

    print("─" * 60, end="\n\n")
