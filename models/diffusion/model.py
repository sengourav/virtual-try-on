# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).to(timesteps.device)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), "constant", 0)
    return emb


def extract(a, t, x_shape):
    out = a.gather(0, t).to(t.device)
    return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = x + self.mha(self.ln(x), self.ln(x), self.ln(x))[0]
        x = x + self.ff_self(x)
        x = x.transpose(1, 2).view(-1, self.channels, *size)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.time_mlp = nn.Linear(time_channels, out_channels) if time_channels is not None else None

    def forward(self, x, time_emb=None):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        if time_emb is not None and self.time_mlp is not None:
            h += self.time_mlp(F.silu(time_emb)).view(-1, h.shape[1], 1, 1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class UNet(nn.Module):
    def __init__(self, in_channels=6, model_channels=128, out_channels=3, num_res_blocks=2,
                 attention_resolutions=(8, 16), channel_mult=(1, 2, 4, 8), time_embed_dim=256):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        self.down_blocks, self.up_blocks = nn.ModuleList(), nn.ModuleList()
        channels, now_channels = [model_channels], model_channels

        for i, mult in enumerate(channel_mult):
            out_channels = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(now_channels, out_channels, time_embed_dim))
                now_channels = out_channels
                channels.append(now_channels)
            if i in attention_resolutions:
                self.down_blocks.append(AttentionBlock(now_channels))
            if i != len(channel_mult) - 1:
                self.down_blocks.append(nn.Conv2d(now_channels, now_channels, 3, stride=2, padding=1))
                channels.append(now_channels)

        self.middle_block = nn.ModuleList([
            ResidualBlock(now_channels, now_channels, time_embed_dim),
            AttentionBlock(now_channels),
            ResidualBlock(now_channels, now_channels, time_embed_dim)
        ])

        for i, mult in reversed(list(enumerate(channel_mult))):
            out_channels = model_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResidualBlock(channels.pop() + now_channels, out_channels, time_embed_dim)
                )
                now_channels = out_channels
            if i in attention_resolutions:
                self.up_blocks.append(AttentionBlock(now_channels))
            if i != 0:
                self.up_blocks.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(now_channels, now_channels, 3, padding=1)
                ))

        self.out = nn.Sequential(
            nn.GroupNorm(32, now_channels),
            nn.SiLU(),
            nn.Conv2d(now_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, time):
        time_emb = self.time_embed(get_timestep_embedding(time, self.time_embed_dim))
        h, hs = self.input_conv(x), []
        for module in self.down_blocks:
            h = module(h, time_emb) if isinstance(module, ResidualBlock) else module(h)
            hs.append(h)
        for module in self.middle_block:
            h = module(h, time_emb) if isinstance(module, ResidualBlock) else module(h)
        for module in self.up_blocks:
            if isinstance(module, ResidualBlock):
                h = module(torch.cat([h, hs.pop()], dim=1), time_emb)
            else:
                h = module(h)
        return self.out(h)


class DiffusionModel(nn.Module):
    def __init__(self, model, beta_start=1e-4, beta_end=0.02, num_timesteps=1000):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        return extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 + \
               extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise

    def p_losses(self, x_0, condition, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise=noise)
        model_input = torch.cat([condition, x_t], dim=1)
        predicted_noise = self.model(model_input, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, condition, x_t, t):
        model_input = torch.cat([condition, x_t], dim=1)
        predicted_noise = self.model(model_input, t)
        mean = extract(torch.sqrt(1.0 / self.alphas_cumprod), t, x_t.shape) * (
            x_t - extract(self.betas, t, x_t.shape) * predicted_noise /
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        )
        noise = torch.zeros_like(x_t) if (t == 0).any() else torch.randn_like(x_t)
        return mean + torch.sqrt(extract(self.posterior_variance, t, x_t.shape)) * noise

    @torch.no_grad()
    def sample(self, condition, shape, device):
        x = torch.randn(shape, device=device)
        for t_idx in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
            x = self.p_sample(condition, x, t)
        return x
