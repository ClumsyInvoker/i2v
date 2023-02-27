import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

import numpy as np
from tqdm import tqdm

from stage2_cINN.AE.modules.AE import BigAE, ResnetEncoder
from omegaconf import OmegaConf

def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]

    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))

    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class simple_denoise_model(nn.Module):
    def __init__(self, in_channels, condition_dim):
        super().__init__()

        t_dim = 16
        self.time_pos_emb = SinusoidalPosEmb(t_dim)
        self.mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4),
            Mish(),
            nn.Linear(t_dim * 4, t_dim)
        )

        input_dim = in_channels + condition_dim + t_dim

        self.fc1 = nn.Linear(input_dim, input_dim*2)
        self.fc2 = nn.Linear(input_dim*2, in_channels)

    def forward(self, x, t):
        t = self.time_pos_emb(t)
        t = self.mlp(t)

        x = torch.cat((x, t), dim=-1)

        x = self.fc1(x)
        x = self.fc2(x)
        return x

class GaussianDiffusion(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.in_channels = kwargs["in_channels"]
        self.condition_dim = kwargs["condition_dim"]
        denoise_model = kwargs["denoise_model"]
        timesteps = kwargs["timesteps"]
        loss_type = kwargs["loss_type"]
        betas = kwargs["betas"]
        self.control = kwargs["control"]

        # self.image_size = image_size
        self.denoise_model = denoise_model if denoise_model is not None else simple_denoise_model(self.in_channels, self.condition_dim)

        if betas is not None:
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        dic = kwargs['dic']
        model_path = dic['model_path'] + dic['model_name'] + '/'
        config = OmegaConf.load(model_path + 'config_stage2_AE.yaml')
        self.embedder = ResnetEncoder(config.AE).cuda()
        self.embedder.load_state_dict(torch.load(model_path + dic['checkpoint_name'] + '.pth')['state_dict'])
        _ = self.embedder.eval()

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised = True, condition_tensor=None):
        denoise_model_input = x
        if condition_tensor is not None:
            denoise_model_input = broadcat((condition_tensor, x), dim=1)

        denoise_model_output = self.denoise_model(denoise_model_input, t)

        x_recon = self.predict_start_from_noise(x, t=t, noise=denoise_model_output)

        if clip_denoised:
            x_recon.clamp_(0., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised = True, repeat_noise=False, condition_tensor=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, condition_tensor=condition_tensor)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, condition_tensor = None):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), condition_tensor = condition_tensor)

        return img

    @torch.no_grad()
    def sample(self, batch_size=16, condition_tensor=None):
        assert not (self.condition_dim > 0 and not condition_tensor is not None), 'the conditioning tensor needs to be passed'

        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), condition_tensor=condition_tensor)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = t if t is not None else self.num_timesteps - 1

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        # noise = default(noise, lambda: torch.randn_like(x_start))
        noise = noise if noise is not None else torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise=None, condition_tensor=None):
        # b, c, h, w = x_start.shape
        noise = noise if noise is not None else torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if condition_tensor is not None:
            x_noisy = broadcat((condition_tensor, x_noisy), dim=1)

        x_recon = self.denoise_model(x_noisy, t)

        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, xc=None, reverse=False, *args, **kwargs):
        b = x.size(0)
        device = x.device

        with torch.no_grad():
            embed = self.embedder.encode(xc[0]).mode().reshape(b, -1).detach()
            embed = torch.cat((embed, self.embed_pos(xc[1])), dim=1) if self.control else embed

        if reverse:
            return self.reverse(x, embed)

        # x_start = torch.cat([x, embed], dim=-1) if xc is not None else x
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.get_x_recon_and_noise(x, t, condition_tensor=embed, *args, **kwargs)

    def get_x_recon_and_noise(self, x_start, t, noise=None, condition_tensor=None):
        noise = noise if noise is not None else torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if condition_tensor is not None:
            # x_noisy = broadcat((condition_tensor, x_noisy), dim=1)
            x_noisy = torch.cat((condition_tensor, x_noisy), dim=-1)

        x_recon = self.denoise_model(x_noisy, t)
        return x_recon, noise

    def reverse(self, x_t, condition_tensor=None):
        b = x_t.size(0)
        device = x_t.device

        denoise_model_input = x_t
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        if condition_tensor is not None:
            # denoise_model_input = broadcat((condition_tensor, x), dim=1)
            denoise_model_input = torch.cat((condition_tensor, denoise_model_input), dim=-1)

        denoise_model_output = self.denoise_model(denoise_model_input, t)

        x_recon = self.predict_start_from_noise(x_t, t=t, noise=denoise_model_output)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x_t, t=t)
        return model_mean

    def embed_pos(self, pos):
        pos = pos * self.cond_size - 1e-4
        embed1 = torch.zeros((pos.size(0), self.cond_size))
        embed2 = torch.zeros((pos.size(0), self.cond_size))
        embed3 = torch.zeros((pos.size(0), self.cond_size))
        embed1[np.arange(embed1.size(0)), pos[:, 0].long()] = 1
        embed2[np.arange(embed2.size(0)), pos[:, 1].long()] = 1
        embed3[np.arange(embed3.size(0)), pos[:, 2].long()] = 1
        return torch.cat((embed1, embed2, embed3), dim=1).cuda()