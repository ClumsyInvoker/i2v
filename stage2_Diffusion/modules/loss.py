import torch.nn as nn
import torch.nn.functional as F
import torch
import wandb

class DiffusionLoss(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x_recon, noise, logger, mode='eval'):
        loss = (noise - x_recon).abs().mean()
        L2_loss = F.mse_loss(noise, x_recon)
        loss_dic = {
            f"Loss": loss.item(),
            f"L2_Loss": L2_loss.item(),
        }
        logger.append(loss_dic)
        ## Add description to keys to be identified either as train or eval
        loss_dic = {mode + '_' + key:val for key, val in loss_dic.items()}
        wandb.log(loss_dic)
        return L2_loss


class DiffusionAllLoss(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x_recon, noise, seq_orig, seq_gen, logger, mode='eval'):
        loss = (noise - x_recon).abs().mean()
        L2_loss = F.mse_loss(noise, x_recon)
        L_recon = torch.mean(torch.abs((seq_gen - seq_orig)))
        loss_dic = {
            f"Loss": loss.item(),
            f"L2_Loss": L2_loss.item(),
            f"Recon_Loss": L_recon.item(),
        }
        logger.append(loss_dic)
        ## Add description to keys to be identified either as train or eval
        loss_dic = {mode + '_' + key:val for key, val in loss_dic.items()}
        wandb.log(loss_dic)
        return L2_loss + L_recon
