import torch, numpy as np, os
import torch.nn as nn
from stage2_cINN.modules.flow_blocks import ConditionalFlow
from stage2_cINN.modules.modules import BasicFullyConnectedNet
from stage2_cINN.AE.modules.AE import BigAE, ResnetEncoder
from omegaconf import OmegaConf

def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    assert attn_type in ["vanilla", "vanilla-xformers", "memory-efficient-cross-attn", "linear", "none"], f'attn_type {attn_type} unknown'
    # if XFORMERS_IS_AVAILBLE and attn_type == "vanilla":
    #    attn_type = "vanilla-xformers"
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        assert attn_kwargs is None
        return AttnBlock(in_channels)
    # elif attn_type == "vanilla-xformers":
    #   print(f"building MemoryEfficientAttnBlock with {in_channels} in_channels...")
    #    return MemoryEfficientAttnBlock(in_channels)
    # elif type == "memory-efficient-cross-attn":
    #    attn_kwargs["query_dim"] = in_channels
    #   return MemoryEfficientCrossAttentionWrapper(**attn_kwargs)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        raise NotImplementedError()



class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


class UNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        dim = kwargs["dim"]
        channels = kwargs["channels"]
        condition_dim = kwargs["condition_dim"]
        dim_mults = kwargs["dim_mults"] if "flow_embedding_channels" in kwargs else (1, 2, 4, 8)

        input_channels = channels + condition_dim
        dims = [input_channels, *map(lambda m: dim * m, dim_mults)]

        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )

        # First Frame Encoder
        dic = kwargs['dic']
        model_path = dic['model_path'] + dic['model_name'] + '/'
        config = OmegaConf.load(model_path + 'config_stage2_AE.yaml')
        self.embedder = ResnetEncoder(config.AE).cuda()
        self.embedder.load_state_dict(torch.load(model_path + dic['checkpoint_name'] + '.pth')['state_dict'])
        _ = self.embedder.eval()


class SupervisedTransformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        in_channels = kwargs["flow_in_channels"]
        mid_channels = kwargs["flow_mid_channels"]
        hidden_depth = kwargs["flow_hidden_depth"]
        n_flows = kwargs["n_flows"]
        conditioning_option = kwargs["flow_conditioning_option"]
        embedding_channels = (
            kwargs["flow_embedding_channels"]
            if "flow_embedding_channels" in kwargs
            else kwargs["flow_in_channels"]
        )

        self.control = kwargs["control"]
        self.cond_size = 10 if self.control else 0

        self.flow = ConditionalFlow(
            in_channels=in_channels,
            embedding_dim=embedding_channels + self.cond_size*3,
            hidden_dim=mid_channels,
            hidden_depth=hidden_depth,
            n_flows=n_flows,
            conditioning_option=conditioning_option,
            control=self.control
        )

        dic = kwargs['dic']
        model_path = dic['model_path'] + dic['model_name'] + '/'
        config = OmegaConf.load(model_path + 'config_stage2_AE.yaml')
        self.embedder = ResnetEncoder(config.AE).cuda()
        self.embedder.load_state_dict(torch.load(model_path + dic['checkpoint_name'] + '.pth')['state_dict'])
        _ = self.embedder.eval()

    def sample(self, shape, cond):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde, cond)
        return sample

    def embed_pos(self, pos):
        pos = pos * self.cond_size - 1e-4
        embed1 = torch.zeros((pos.size(0), self.cond_size))
        embed2 = torch.zeros((pos.size(0), self.cond_size))
        embed3 = torch.zeros((pos.size(0), self.cond_size))
        embed1[np.arange(embed1.size(0)), pos[:, 0].long()] = 1
        embed2[np.arange(embed2.size(0)), pos[:, 1].long()] = 1
        embed3[np.arange(embed3.size(0)), pos[:, 2].long()] = 1
        return torch.cat((embed1, embed2, embed3), dim=1).cuda()

    def forward(self, input, cond, reverse=False, train=False):

        with torch.no_grad():
            embed = self.embedder.encode(cond[0]).mode().reshape(input.size(0), -1).detach()
            embed = torch.cat((embed, self.embed_pos(cond[1])), dim=1) if self.control else embed

        if reverse:
            return self.reverse(input, embed)

        out, logdet = self.flow(input, embed)

        return out, logdet

    def reverse(self, out, cond):
        return self.flow(out, cond, reverse=True)


