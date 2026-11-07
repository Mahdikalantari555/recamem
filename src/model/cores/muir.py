"""Experimental sequential CPU Muir mixer, based on arXiv:2611.08376, eq. 8.

This is a transparent PyTorch reference, not the paper's TPU chunkwise kernel.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.model.cores.contracts import CoreOutput, StateSpec
from src.model.cores.diagonal_cores import SwiGLUBlock
from src.model.cores.matrix_cores import _StatefulCausalConv, _l2norm


class MuirLayer(nn.Module):
    def __init__(self, d, heads, d_ff, kernel, readout="dynamic", ranks=None):
        super().__init__()
        self.d, self.heads, self.dim = d, heads, d // heads
        self.readout = readout
        # Scaled-down 1:1:2:2:2 ranks; unlike the paper these are independent
        # low-rank up-projections, for readability on CPU.
        small, large = max(1, d // 16), max(2, d // 8)
        self.ranks = tuple(ranks) if ranks is not None else (small, small, large, large, large)
        if len(self.ranks) != 5 or any(not isinstance(r, int) or r < 1 or r > d for r in self.ranks):
            raise ValueError("Muir requires five gate ranks in [1, d_model]")
        self.pre_norm = nn.RMSNorm(d, eps=1e-6)
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.conv = _StatefulCausalConv(3*d, kernel)
        self.down = nn.Linear(d, sum(self.ranks), bias=False)
        self.up = nn.ModuleList([nn.Linear(r, d, bias=(i in (0, 1))) for i, r in enumerate(self.ranks)])
        self.a_log = nn.Parameter(torch.zeros(heads, self.dim))
        self.gamma = nn.Parameter(torch.ones(heads, self.dim))
        self.out_norm = nn.RMSNorm(self.dim, eps=1e-6)
        self.out = nn.Linear(d, d, bias=False)
        self.ffn = SwiGLUBlock(d, d_ff)

    def forward(self, x, matrix, cache):
        batch, length, _ = x.shape
        u = self.pre_norm(x)
        qkv, cache = self.conv.forward_with_cache(self.qkv(u), cache)
        q, k, v = F.silu(qkv).chunk(3, dim=-1)
        shape = (batch, length, self.heads, self.dim)
        q, k, v = _l2norm(q.reshape(shape)), _l2norm(k.reshape(shape)), v.reshape(shape)
        gates = [up(part).reshape(shape) for up, part in zip(self.up, self.down(u).split(self.ranks, -1))]
        f, b, g, w, gain = gates
        alpha = torch.exp(-torch.exp(self.a_log) * F.softplus(f))
        b, w = 2 * torch.sigmoid(b), torch.sigmoid(w)
        gamma = self.gamma * (2 * torch.sigmoid(gain) if self.readout == "dynamic" else 1)
        output = []
        for t in range(length):
            decayed = alpha[:, t, :, :, None] * matrix
            prediction = ((b[:, t] * k[:, t]).unsqueeze(-2) @ decayed).squeeze(-2)
            matrix = decayed + k[:, t, :, :, None] * (w[:, t] * v[:, t] - prediction).unsqueeze(-2)
            output.append(((q[:, t] * gamma[:, t] if self.readout == "dynamic" else q[:, t] * gamma).unsqueeze(-2) @ matrix).squeeze(-2))
        y = torch.stack(output, dim=1)
        y = (self.out_norm(y) * F.silu(g)).reshape(batch, length, self.d)
        return self.ffn(x + self.out(y)), matrix, cache


class MuirCore(nn.Module):
    core_id = "C7"
    supports_online_rtrl = False

    def __init__(self, d_model, n_layers, n_heads, d_ff, conv_kernel=4, readout="dynamic", gate_ranks=None, **_):
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads or n_layers <= 0 or conv_kernel < 1:
            raise ValueError("invalid Muir dimensions")
        if readout not in ("static", "dynamic"):
            raise ValueError("readout must be static or dynamic")
        self.d_model, self.n_layers, self.n_heads = d_model, n_layers, n_heads
        self.dim, self.kernel = d_model // n_heads, conv_kernel
        self.layers = nn.ModuleList([MuirLayer(d_model, n_heads, d_ff, conv_kernel, readout, gate_ranks) for _ in range(n_layers)])
        self._persistent_state = None

    def state_spec(self, batch_size=1, seq_len=1):
        del batch_size, seq_len
        count = self.n_layers * (self.n_heads * self.dim**2 + 3*self.d_model*(self.kernel-1))
        return StateSpec("constant_matrix", count*4, (self.n_heads, self.dim, self.dim), self.n_layers)

    def init_state(self, batch_size=1, device=None, dtype=torch.float32):
        device = device or next(self.parameters()).device
        return {"matrix": torch.zeros(self.n_layers, batch_size, self.n_heads, self.dim, self.dim, device=device, dtype=dtype),
                "conv": torch.zeros(self.n_layers, batch_size, 3*self.d_model, self.kernel-1, device=device, dtype=dtype)}

    def reset_state(self):
        self._persistent_state = None

    def forward_sequence(self, x, state=None, token_ids=None, layer_hook=None):
        if x.shape[1] == 0:
            raise ValueError("Muir requires a nonempty sequence")
        if state is None:
            state = self.init_state(x.shape[0], x.device, x.dtype)
        h, matrices, caches = x, [], []
        for i, layer in enumerate(self.layers):
            h, m, c = layer(h, state["matrix"][i], state["conv"][i])
            if layer_hook is not None:
                h = layer_hook(i, h, token_ids)
            matrices.append(m); caches.append(c)
        result = {"matrix": torch.stack(matrices), "conv": torch.stack(caches)}
        self._persistent_state = {k: v.detach() for k, v in result.items()}
        return CoreOutput(h, result, self.state_spec())

    def step(self, x_t, state=None, token_id=None, layer_hook=None):
        return self.forward_sequence(x_t.unsqueeze(1), state, None if token_id is None else token_id.unsqueeze(1), layer_hook)
