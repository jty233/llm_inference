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
        # 平铺到序列形式 [batch, seq_len, embed_dim]
        attn_output, _ = self.mha(x, x, x)  # [batch, 784, embed_dim]
        out = self.classifier(attn_output)     # [batch, num_classes]
        return out

# 超参数
batch_size = 512
epochs = 40
learning_rate = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 模型、损失、优化器
model = MNISTWithMHA().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0
    for data, target in tqdm.tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

    # 在测试集上评估
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
    acc = correct / len(test_loader.dataset)
    print(f'Test Accuracy: {acc:.4f}')

print('Training complete')
# 获取模型权重
state_dict = model.state_dict()

# 保存为 .safetensors
save_file(state_dict, "mnist_mha.safetensors")