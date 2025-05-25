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

# ————————————————————————————————
# 4. 准备输入并跑一次推理
# ————————————————————————————————
text = "<|im_start|>user\nWhat can you do?<|im_end|>\n<|im_start|>assistant\n<think>\n"
# text = "<|im_start|>user"
inputs = tokenizer(text, return_tensors="pt").to(device)
print(inputs)

# exit(0)

with torch.no_grad():
    outputs = model(**inputs)          # outputs 是一个 ModelOutput，包含 logits
    logits  = outputs.logits           # 形状 [batch_size, seq_len, vocab_size]
    
    # 只拿最后一个时刻（即预测下一个 token）的 logits
    # next_token_logits 形状 [batch_size, vocab_size]
    next_token_logits = logits[:, -1, :]
    
    # 对 logits 做 softmax 得到概率分布
    probs = torch.softmax(next_token_logits, dim=-1)  # 形状 [batch_size, vocab_size]

    # 假设我们想要前 top_k 个概率
    top_k = 40
    topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)
    # topk_probs: 形状 [batch_size, top_k]
    # topk_indices: 形状 [batch_size, top_k]

    # 把 token_id 转成可读的字符串
    # 因为 batch_size=1，所以可以先 squeeze 掉 batch 维度
    topk_probs = topk_probs.squeeze(0).cpu().numpy()     # [top_k]
    topk_ids   = topk_indices.squeeze(0).cpu().tolist()   # [top_k]
    topk_tokens = tokenizer.convert_ids_to_tokens(topk_ids)

    # 打印一下结果
    for idx, (id, tok, prob) in enumerate(zip(topk_ids,topk_tokens, topk_probs)):
        print(f"Rank {idx+1:2d}: Id = {id:<8} Token = {tok:<12}  Probability = {prob:.6f}")

exit(0)

# ————————————————————————————————
# 5. 打印 intermediates 里存储的张量“数值示例”
# ————————————————————————————————
for module, arr in intermediates.items():
    print("─" * 60)
    print(f"模块: {module}")
    print(f"张量形状: {arr.shape}")
    if module in pre_inputs:
        print("运算前：")
        flat = pre_inputs[module][0, -1].flatten()
        print("前 10 个元素(展平后)：", flat[:10].tolist())
        print("后 10 个元素(展平后)：", flat[-10:].tolist())
        print('运算后:')
    flat = arr[0, -1].flatten()
    print("前 10 个元素(展平后)：", flat[:10].tolist())
    print("后 10 个元素(展平后)：", flat[-10:].tolist())
    arr = pre_inputs.get(module)

    print("─" * 60, end="\n\n")
