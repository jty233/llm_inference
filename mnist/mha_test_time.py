import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tqdm
from safetensors.torch import save_file

# 定义使用多头注意力的模型
class MNISTWithMHA(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_classes=10):
        super(MNISTWithMHA, self).__init__()
        # 将28x28图像展平到序列长度784，嵌入到embed_dim维
        self.embedding = nn.Linear(28*28, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [batch, 1, 28, 28]
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # [batch, 784]
        x = self.embedding(x)       # [batch, embed_dim]
        x = x.unsqueeze(1)
        # 平铺到序列形式 [batch, seq_len, embed_dim]
        attn_output, _ = self.mha(x, x, x)  # [batch, 1, embed_dim]
        attn_output = attn_output.squeeze(1)
        out = self.classifier(attn_output)     # [batch, num_classes]
        return out
    
from time import time
model = MNISTWithMHA()
x = torch.randn((100000,1,28,28))
t_start = time()
model(x)
print(f"numpy took {time()-t_start:.3f}s")
