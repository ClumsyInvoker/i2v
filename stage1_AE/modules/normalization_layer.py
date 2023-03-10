import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class Spade(nn.Module):
    def __init__(self, num_features, num_groups=16):
        super().__init__()
        self.num_features = num_features
        while self.num_features % num_groups != 0:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, num_features, affine=False)

        self.conv       = nn.Conv2d(3, 128, 3, 1, 1)
        self.conv_gamma = nn.Conv2d(128, num_features, 3, 1, 1)
        self.conv_beta  = nn.Conv2d(128, num_features, 3, 1, 1)
        self.activate   = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, y):
        normalized = self.norm(x)
        y = F.interpolate(y, mode='bilinear', size=x.shape[-2:], align_corners=True).cuda()
        y = self.activate(self.conv(y))
        gamma = self.conv_gamma(y).unsqueeze(2).repeat_interleave(x.size(2), 2)
        beta  = self.conv_beta(y).unsqueeze(2).repeat_interleave(x.size(2), 2)
        return normalized * (1 + gamma) + beta


class Norm3D(nn.Module):
    def __init__(self, num_features, num_groups=16):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.GroupNorm(num_groups, num_features, affine=True)

    def forward(self, x):
        out = self.bn(x)
        return out


class ADAIN(nn.Module):
    def __init__(self, num_features, z_dim):
        super().__init__()
        self.num_features = num_features
        self.norm = nn.InstanceNorm3d(num_features, affine=False, track_running_stats=False)

        self.linear = nn.Linear(z_dim, num_features*2)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, y):
        out = self.norm(x)
        gamma, beta = self.linear(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1, 1) * out + beta.view(-1, self.num_features, 1, 1, 1)
        return out


class ADAIN3D(nn.Module):
    def __init__(self, num_features, z_dim):
        super().__init__()
        self.num_features = num_features
        self.norm = nn.InstanceNorm3d(num_features, affine=False, track_running_stats=False)

        # self.linear = nn.Linear(z_dim, num_features*2)
        self.conv = nn.Conv3d(z_dim, num_features, 1, 1, 0)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 5)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, x, y):
        out = self.norm(x)
        # gamma, beta = self.linear(y).chunk(2, 1)
        y = self.conv(y)
        y_mean, y_std = self.calc_mean_std(y)
        out = y_std.view(-1, self.num_features, 1, 1, 1) * out + y_mean.view(-1, self.num_features, 1, 1, 1)
        return out
