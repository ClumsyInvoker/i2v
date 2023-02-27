import torch
import torch.nn as nn
import math
from abc import abstractmethod
from einops import rearrange

from stage2_Diffusion.modules.util import normalization
from stage2_Diffusion.modules.Attention import BasicTransformerBlock


try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
    print("No module 'xformers'. Proceeding without it.")



def exists(x):
    return x is not None

# 有val时返回val，val为None时返回d
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# 残差模块，将输入加到输出上
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

# 输入自动化匹配
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

# 上采样（反卷积）
#def Upsample(channel):
#    return nn.ConvTranspose2d(channel, channel, 4, 2, 1)
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, use_transpose=False):
        super().__init__()
        self.with_conv = with_conv
        self.use_transpose = use_transpose
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        if self.use_transpose:
            self.conv = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1)

    def forward(self, x):
        # 使用反卷积上采样
        if self.use_transpose:
            return self.conv(x)

        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

# 下采样
#def Downsample(channel):
#    return nn.Conv2d(channel, channel, 4, 2, 1)
class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def get_timestep_embedding(timesteps, embedding_dim):
        device = timesteps.device
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


class ResnetBlock(TimestepBlock):
    def __init__(self, in_channels, emb_channels, dropout, out_channels=None, use_conv_shortcut=False):
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.out_channels = in_channels if out_channels is None else out_channels
        self.in_layers = nn.Sequential(
                normalization(in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, self.out_channels, 3, padding=1))

        self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_channels, self.out_channels))

        self.out_layers = nn.Sequential(
                normalization(self.out_channels),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))

        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            if use_conv_shortcut:
                self.skip_connection = nn.Conv2d(in_channels, self.out_channels, 3, 1, 1)
            else:
                self.skip_connection = nn.Conv2d(in_channels, self.out_channels, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        if emb is not None:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            h = h + emb_out
        h = self.out_layers(h)

        x = self.skip_connection(x)

        return h + x


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = normalization(in_channels)

        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1,stride=1,padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)])

        self.proj_out = nn.Conv2d(inner_dim,in_channels,kernel_size=1,stride=1,padding=0)


    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in



class UNet(nn.Module):
    def __init__(self, resolution, in_ch, ch, out_ch, num_res_blocks,
                 attn_resolutions, context_dim=None, dropout=0.0, ch_mult=(1,2,4,8), num_head=8, transformer_depth=1,
                 resamp_with_conv=True, use_timestep=True, use_linear_attn=False, attn_type="vanilla"):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_ch = in_ch

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                nn.Linear(self.ch,
                                self.temb_ch),
                nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])
            self.temb.nonlinearity = nn.SiLU()

        # input block
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(in_ch, ch, 3, padding=1))]
        )

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_ch,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(TimestepEmbedSequential(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         emb_channels=self.temb_ch,
                                         dropout=dropout)))
                block_in = block_out
                if curr_res in attn_resolutions:
                    # attn.append(make_attn(block_in, attn_type=attn_type))
                    dim_head = block_in // num_head
                    attn.append(TimestepEmbedSequential(
                        SpatialTransformer(block_in, num_head, dim_head, depth=transformer_depth, context_dim=context_dim)))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = TimestepEmbedSequential(ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       emb_channels=self.temb_ch,
                                       dropout=dropout))
        dim_head = block_in // num_head
        # self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.attn_1 = TimestepEmbedSequential(SpatialTransformer(block_in, num_head, dim_head, depth=transformer_depth, context_dim=context_dim))
        self.mid.block_2 = TimestepEmbedSequential(ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       emb_channels=self.temb_ch,
                                       dropout=dropout))

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(TimestepEmbedSequential(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         emb_channels=self.temb_ch,
                                         dropout=dropout)))
                block_in = block_out
                if curr_res in attn_resolutions:
                    # attn.append(make_attn(block_in, attn_type=attn_type))
                    dim_head = block_in // num_head
                    attn.append(TimestepEmbedSequential(SpatialTransformer(block_in, num_head, dim_head, depth=transformer_depth,
                                                   context_dim=context_dim)))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = normalization(block_in)
        self.nonlinaerity_out = nn.SiLU()
        self.conv_out = nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

        self.out = nn.Sequential(self.norm_out, self.nonlinaerity_out, self.conv_out)

    def forward(self, x, t=None, context=None):
        # print(x.shape)
        assert x.shape[2] == x.shape[3] == self.resolution
        # if context is not None:
            # assume aligned context, cat along channel axis
            # x = torch.cat((x, context), dim=1)

        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = self.temb.nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb, context)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h, temb, context)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb, context)
        h = self.mid.attn_1(h, temb, context)
        h = self.mid.block_2(h, temb, context)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb, context)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, temb, context)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.out(h)

        return h

    def get_last_layer(self):
        return self.out.conv_out.weight


class UNet_1D(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        dim = kwargs["dim"]
        in_channels = kwargs["in_channels"]
        condition_dim = kwargs["condition_dim"]

        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UNetDown(n_feat, n_feat)
        self.down2 = UNetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)

        self.contextembed1 = EmbedFC(n_classes, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1 * n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),  # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UNetUp(4 * n_feat, n_feat)
        self.up2 = UNetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) input, c is context feature, t is timestep,
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)

        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1, self.n_classes)
        context_mask = (-1 * (1 - context_mask))  # need to flip 0 <-> 1
        c = c * context_mask

        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up1(cemb1 * up1 + temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


