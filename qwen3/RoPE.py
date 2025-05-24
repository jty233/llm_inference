import torch
import torch.nn as nn
import math

class RoPE:
    """
    Rotary Positional Embedding 模块，提供一个静态方法来对任意形状的 tensor 应用 RoPE。
    """

    @staticmethod
    def apply_rope(x: torch.Tensor):
        """
        对输入向量 x 应用 RoPE 旋转。
        
        Args:
            x: shape = [batch_size, seq_len, dim], dim 必须为偶数。
        
        Returns:
            x_rotated: 同样形状的 tensor，已注入位置信息。
        """
        batch_size, seq_len, dim = x.shape
        assert dim % 2 == 0, "RoPE 要求 dim 必须为偶数。"

        # 1. 计算每个维度对应的频率：omega = 1 / (10000^{2k/d})
        #    k = 0, 1, ..., dim/2 - 1
        half_dim = dim // 2
        # torch.arange(half_dim) 生成 [0, 1, ..., half_dim-1]
        inv_freq = 1.0 / (1e6 ** (2 * torch.arange(half_dim, dtype=torch.float32) / dim))  # [half_dim]
        # inv_freq 形状：[half_dim]

        # 2. 构造每个位置 p 的角度矩阵 [seq_len, half_dim]
        #    pos = torch.arange(seq_len).unsqueeze(1)  => [seq_len, 1]
        #    angles = pos * inv_freq  => [seq_len, half_dim]
        pos = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1)  # [seq_len, 1]
        angles = pos * inv_freq.to(x.device)  # [seq_len, half_dim]

        # 3. 计算 cos 和 sin：[seq_len, half_dim]
        cos = torch.cos(angles)  # [seq_len, half_dim]
        sin = torch.sin(angles)  # [seq_len, half_dim]

        # 4. 将 x 分成 even 和 odd 两份：x_even = x[..., 0::2], x_odd = x[..., 1::2]
        #    再把它们按 RoPE 公式做旋转。
        x = x.view(batch_size, seq_len, half_dim, 2)  # [batch, seq_len, half_dim, 2]
        x_even = x[..., 0]  # [batch, seq_len, half_dim]
        x_odd  = x[..., 1]  # [batch, seq_len, half_dim]

        # 注意 cos/sin 形状是 [seq_len, half_dim]，需要 broadcast 到 [batch, seq_len, half_dim]
        cos = cos.unsqueeze(0)  # [1, seq_len, half_dim]
        sin = sin.unsqueeze(0)  # [1, seq_len, half_dim]

        # 旋转公式：
        # x'_even = x_even * cos - x_odd * sin
        # x'_odd  = x_even * sin + x_odd * cos
        x_rot_even = x_even * cos - x_odd * sin  # [batch, seq_len, half_dim]
        x_rot_odd  = x_even * sin + x_odd * cos  # [batch, seq_len, half_dim]

        # 5. 把旋转后的 even/odd 再 interleave 回原始维度
        x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1)  # [batch, seq_len, half_dim, 2]
        x_rot = x_rot.view(batch_size, seq_len, dim)  # [batch, seq_len, dim]

        return x_rot

if __name__ == "__main__":
    # 测试示例
    batch_size = 2
    seq_len = 8
    dim = 32
    num_heads = 4

    # 随机输入模拟：batch_size 个序列，每个序列长度为 seq_len，隐藏维度为 dim
    x = torch.arange(0, 12)
    x = x.reshape((1, 2, 6))
    print(x)
    res = RoPE.apply_rope(x)
    print(res)
    new = torch.concat((x, res), 1)
    print(new)
    res = RoPE.apply_rope(new)
    print(res)
