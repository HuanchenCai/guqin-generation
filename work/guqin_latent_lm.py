"""Autoregressive "latent LM" for guqin: SAME-L latent frames in, next frame out.

No text. A causal transformer reads past latent frames (256-d, ~10.8 frames
per second) and a small flow-matching MLP head samples the next frame, in the
spirit of MAR-style continuous autoregression. Context frames are noised
during training so the model tolerates its own imperfect outputs when it runs
for a long time (see arXiv 2411.18447).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 256


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * 1000 * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width))
        self.ada = nn.Linear(width, width * 3)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, c):
        shift, scale, gate = self.ada(c).chunk(3, dim=-1)
        return x + gate * self.mlp(self.norm(x) * (1 + scale) + shift)


class FlowHead(nn.Module):
    """Predicts the flow velocity for one latent frame given the transformer state."""

    def __init__(self, cond_dim: int, width: int = 1024, depth: int = 6):
        super().__init__()
        self.inp = nn.Linear(LATENT_DIM, width)
        self.cond = nn.Linear(cond_dim, width)
        self.time = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(ResBlock(width) for _ in range(depth))
        self.out_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.out = nn.Linear(width, LATENT_DIM)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x_t, t, h):
        c = self.cond(h) + self.time(timestep_embedding(t, 256))
        x = self.inp(x_t)
        for block in self.blocks:
            x = block(x, c)
        return self.out(self.out_norm(x))


class LatentLM(nn.Module):
    def __init__(self, d_model=768, layers=12, heads=12, max_len=1024, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(LATENT_DIM, d_model)
        self.sigma_emb = nn.Linear(1, d_model)
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = FlowHead(d_model)
        self.max_len = max_len

    def hidden(self, context, sigma):
        """context: (B, T, 256) normalised frames -> (B, T+1, d) states.

        State i predicts frame i (state 0 comes from a learned start token)."""
        b, t, _ = context.shape
        x = self.inp(context) + self.sigma_emb(sigma.view(b, 1, 1).to(context.dtype))
        x = torch.cat([self.bos.expand(b, -1, -1), x], dim=1)[:, : self.max_len]
        x = x + self.pos[:, : x.shape[1]]
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device)
        return self.norm(self.blocks(x, mask=mask, is_causal=True))

    def loss(self, frames, sigma_max=0.5, noise_prob=0.5, mul=4):
        """frames: (B, T, 256). Predict every frame from the frames before it."""
        b = frames.shape[0]
        sigma = torch.rand(b, device=frames.device) * sigma_max
        sigma = torch.where(torch.rand(b, device=frames.device) < noise_prob, sigma, torch.zeros_like(sigma))
        noisy = frames + sigma.view(b, 1, 1) * torch.randn_like(frames)
        h = self.hidden(noisy[:, :-1], sigma)                  # (B, T, d)
        target = frames.reshape(-1, LATENT_DIM).repeat(mul, 1)
        cond = h.reshape(-1, h.shape[-1]).repeat(mul, 1)
        # Logit-normal timesteps, as in rectified-flow image models.
        t = torch.sigmoid(torch.randn(target.shape[0], device=frames.device))
        noise = torch.randn_like(target)
        x_t = (1 - t[:, None]) * noise + t[:, None] * target
        v = self.head(x_t, t, cond)
        return F.mse_loss(v.float(), (target - noise).float())

    @torch.no_grad()
    def sample_next(self, h_last, steps=32, temperature=1.0):
        """Euler-integrate the flow from noise to one frame per row of h_last."""
        x = torch.randn(h_last.shape[0], LATENT_DIM, device=h_last.device) * temperature
        for i in range(steps):
            t = torch.full((x.shape[0],), i / steps, device=x.device)
            x = x + self.head(x, t, h_last) / steps
        return x
