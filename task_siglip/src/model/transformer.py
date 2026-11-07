from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()

        if num_heads <= 0 or dim <= 0 or dim % num_heads != 0:
            raise ValueError("dim and num_heads must be positive and dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        for start in range(0, 3 * dim, dim):
            nn.init.xavier_uniform_(self.qkv.weight[start : start + dim])
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, N, D = x.shape

        # 计算 Q, K, V 矩阵
        qkv = self.qkv(x)  # (B, N, 3 * D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Softmax 归一化
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Attention 加权求和
        out = attn @ v

        # 拼接多个 Attention Head
        out = out.transpose(1, 2).contiguous().reshape(B, N, D)

        # 输出投影
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim):
        super().__init__()
        self.self_attention = SelfAttention(dim=embed_dim, num_heads=num_heads)
        fc1 = nn.Linear(embed_dim, mlp_hidden_dim)
        fc2 = nn.Linear(mlp_hidden_dim, embed_dim)
        for layer in (fc1, fc2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.normal_(layer.bias, std=1e-6)
        self.mlp = nn.Sequential(fc1, nn.GELU(approximate="tanh"), fc2)
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))

        return x
