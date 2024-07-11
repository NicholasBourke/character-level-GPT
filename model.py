import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class LayerNorm(nn.Module):
    def __init__(self, norm_dim, eps=1e-05):
        super(LayerNorm, self).__init__()
        self.gain = nn.Parameter(torch.ones(norm_dim))
        self.bias = nn.Parameter(torch.zeros(norm_dim))
        self.eps = eps
        self.num_dims = len(norm_dim)

    def forward(self, x):
        dims = tuple(range(-self.num_dims, 0))
        mu = x.mean(dims, keepdim=True)
        var = x.var(dims, keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps) * self.gain + self.bias
    


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super(MultiHeadAttention, self).__init__()
        assert cfg.D % cfg.n_head == 0
        self.cfg = cfg
        self.Wqkv = nn.Linear(cfg.D, 3*cfg.D, bias=cfg.bias)
        self.Wo = nn.Linear(cfg.D, cfg.D, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x):
        q, k, v = self.Wqkv(x).split(self.cfg.D, dim=-1)
        q = q.view(-1, self.cfg.L, self.cfg.n_head, self.cfg.D//self.cfg.n_head).transpose(1, 2)
        k = k.view(-1, self.cfg.L, self.cfg.n_head, self.cfg.D//self.cfg.n_head).transpose(1, 2)
        v = v.view(-1, self.cfg.L, self.cfg.n_head, self.cfg.D//self.cfg.n_head).transpose(1, 2)

        p = self.dropout if self.training else 0
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=p, is_causal=True)
        attn = attn.transpose(1, 2).reshape(x.size(0), self.cfg.L, self.cfg.D)
        return self.Wo(attn)
    


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super(FeedForward, self).__init__()
        self.W1 = nn.Linear(cfg.D, 4*cfg.D, bias=cfg.bias)
        self.W2 = nn.Linear(4*cfg.D, cfg.D, bias=cfg.bias)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.W2(self.dropout(self.relu(self.W1(x))))
    


class Layer(nn.Module):
    def __init__(self, cfg):
        super(Layer, self).__init__()
        self.pos_embed = nn.Embedding(cfg.L, cfg.D)
        self.attn = MultiHeadAttention(cfg)
        self.nrm_1 = LayerNorm((cfg.L, cfg.D))
        self.ff = FeedForward(cfg)
        self.nrm_2 = LayerNorm((cfg.L, cfg.D))

    def forward(self, x):
        p = torch.arange(x.size(-1)).to(x.device)
        xp = x + self.pos_embed(p)
        a = xp + self.attn(self.nrm_1(xp))
        y = a + self.ff(self.nrm_2(a))
        return y



class CharGPT(nn.Module):
    def __init__(self, cfg):
        super(CharGPT, self).__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.K, cfg.D)
        self.stack = nn.ModuleList([Layer(cfg) for _ in range(cfg.n_layer)])
        self.nrm = LayerNorm((cfg.L, cfg.D))
        self.out = nn.Linear(cfg.D, cfg.K, bias=cfg.bias)

        self.out.weight = self.embed.weight # weight tying
        self.init_weights() # weight initialization

    def forward(self, x_tkn):
        x = self.embed(x_tkn)
        for layer in self.stack:
            x = layer(x)
        return self.out(self.nrm(x))

    def init_weights(self):
        for name, mod in self.named_modules():
            if type(mod) == nn.Embedding:
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
            if type(mod) == nn.Linear:
                if name.endswith("Wo") or name.endswith("W2"):
                    nn.init.normal_(mod.weight, mean=0.0, std=0.02/math.sqrt(2 * self.cfg.n_layer))
                else:
                    nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                if mod.bias is not None: nn.init.zeros_(mod.bias)

