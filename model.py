import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models import regnet_x_400mf, RegNet_X_400MF_Weights


@dataclass
class ModelConfig:
    block_size: int = 128
    n_layer: int = 12
    n_head: int = 8
    dim: int = 256
    # dropout: float = 0.25
    dropout: float = 0  # dropout unsupported on MPS
    bias: bool = False


# transformer stuff copied from karpathy/nanoGPT
class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.dim % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.dim, 3 * config.dim, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.dim
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.dim, 4 * config.dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.dim, config.dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.dim, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.dim, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ImageBackbone(nn.Module):
    def __init__(self, config: ModelConfig, pretrained=False):
        super().__init__()

        weights = RegNet_X_400MF_Weights if pretrained else None
        self.model = regnet_x_400mf(weights=weights)  # model dim is 400

        self.model.fc = Identity()  # remove classification layer
        self.fc = nn.Linear(400, config.dim)  # up-project to the model dim

    def forward(self, x):
        out = self.model(x)
        return self.fc(out)


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.pe = nn.Embedding(config.block_size, config.dim)
        self.drop = nn.Dropout(config.dropout)
        self.hidden = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln = LayerNorm(config.dim, bias=config.bias)

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        device = idx.device
        b, t, dim = idx.size()

        assert dim == self.config.dim
        assert t <= self.config.block_size

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.pe(pos)

        x = self.drop(idx + pos_emb)
        for block in self.hidden:
            x = block(x)

        x = self.ln(x)
        return x


class CalibrationModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.image_backbone = ImageBackbone(config, pretrained=False)
        self.transformer = Transformer(config)

        self.means = nn.Linear(config.dim, 2, bias=True)
        self.logvars = nn.Linear(config.dim, 2, bias=True)

        with torch.no_grad():
            nn.init.normal_(self.means.weight, std=0.02)
            nn.init.zeros_(self.means.bias)

            nn.init.normal_(self.logvars.weight, std=0.02)
            nn.init.constant_(self.logvars.bias, -1)  # start with high uncertainty

    def forward(self, x):
        b, t, c, w, h = x.size()
        x = x.view(b * t, c, w, h)
        x = self.image_backbone(x)
        x = x.view(b, t, self.config.dim)
        x = self.transformer(x)

        means = self.means(x)
        logvars = self.logvars(x)

        return means, logvars

    def forward_backbone(self, x):
        x = self.image_backbone(x)
        return x

    def forward_transformer(self, x):
        x = self.transformer(x)
        x = self.means(x)
        return x

    def loss(self, means, logvars, targets, eps=1e-6):
        nll = (
            (0.5 * math.log(2 * math.pi))
            + (0.5 * logvars)
            + (0.5 * ((targets - means) ** 2) / (torch.exp(logvars) + eps))
        )
        return nll.mean()


if __name__ == "__main__":
    model = CalibrationModel(config=ModelConfig)
    print(f"params: {sum(p.numel() for p in model.parameters())}")

    sample_input = torch.randn(16, 8, 3, 120, 90)
    result: torch.Tensor = model(sample_input)

    print(result.shape)
