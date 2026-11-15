import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, seq_len, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, d_model: int, ratio: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ratio * d_model),
            nn.GELU(),
            nn.Linear(ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class IVANano(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.vocab_size = int(config["vocab_size"])
        self.context_length = int(config["context_length"])
        d_model = int(config["d_model"])
        dropout = float(config.get("dropout", 0.0))
        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        self.position_embedding = nn.Embedding(self.context_length, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                d_model, int(config["n_heads"]),
                int(config["mlp_ratio"]), dropout,
            )
            for _ in range(int(config["n_layers"]))
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        if config.get("tie_embeddings", True):
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        batch, seq_len = input_ids.shape
        if seq_len > self.context_length:
            raise ValueError(f"Sequence length {seq_len} exceeds context {self.context_length}")
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.drop(self.token_embedding(input_ids) + self.position_embedding(positions))
        logits = self.lm_head(self.final_norm(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
