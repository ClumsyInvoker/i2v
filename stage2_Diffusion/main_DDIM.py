import numpy as np, os, time, gc
import torch, random
from tqdm import tqdm, trange
from omegaconf import OmegaConf
import argparse
from datetime import datetime
import wandb

from data.get_dataloder import get_loader
import stage2_Diffusion.modules.loss as loss
import stage2_Diffusion.modules.DDIM_UNet as DDIM
import utils.auxiliaries as aux
from stage1_AE.modules import decoder as net
from stage1_AE.modules.resnet3D import Encoder
from metrics.PyTorch_FVD import FVD_logging
from metrics.DTFVD import DTFVD_Score

"""=========================Trainer Function==================================================="""

def trainer(model, encoder, epoch, data_loader, logger, optimizer, loss_func, opt):

    _ = model.train()
    logger.reset()
    data_iter = tqdm(data_loader, position=2)
    inp_string = 'Epoch {} || Loss: --- '.format(epoch)
    data_iter.set_description(inp_string)
    for image_idx, file_dict in enumerate(data_iter):

        seq = file_dict["seq"].type(torch.FloatTensor).cuda()

        post, mean, *_ = encoder(seq[:, 1:].transpose(1, 2))
        cond = [seq[:, 0]] if not opt.Training['control'] else [seq[:, 0],  file_dict["cond"]]
        x_recon, noise = model(post.reshape(post.size(0), -1).detach(), cond)
        loss = loss_func(x_recon, noise, logger, mode='train')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if image_idx % 20 == 0:
            inp_string = 'Epoch {} || Loss: {}'.format(epoch, np.round(loss.item(), 3))
            data_iter.set_description(inp_string)

    ### Empty GPU cache
    torch.cuda.empty_cache()

"""===========================Validation Function==================================================="""

def validator(model, encoder, epoch, data_loader, logger, loss_func, opt):
    _ = model.eval()
    logger.reset()
    data_iter = tqdm(data_loader, position=2)
    inp_string = 'Epoch {} || Loss: --- '.format(epoch)
    data_iter.set_description(inp_string)
    with torch.no_grad():
        for image_idx, file_dict in enumerate(data_iter):

            seq = file_dict["seq"].type(torch.FloatTensor).cuda()

            post, mean, *_ = encoder(seq[:, 1:].transpose(1, 2))
            cond = [seq[:, 0]] if not opt.Training['control'] else [seq[:, 0], file_dict["cond"]]

            x_recon, noise = model(post.reshape(post.size(0), -1).detach(), cond)
            loss = loss_func(x_recon, noise, logger, mode='eval')

            if image_idx % 20 == 0:
                inp_string = 'Epoch {} || Loss: {}'.format(epoch, np.round(loss.item(), 3))
                data_iter.set_description(inp_string)

    ### Empty GPU cache
    torch.cuda.empty_cache()


def main(opt):
    """================= Create Model, Optimizer and Scheduler =========================="""

    ### Load first stage model (Decoder + Encoder)
    model_path = opt.First_stage_model['model_path'] + opt.First_stage_model['model_name'] + '/'
    config = OmegaConf.load(model_path + 'config_stage1.yaml')

    checkpoint_name = model_path + opt.First_stage_model['checkpoint_decoder'] + '.pth'
    generator = net.Generator(config.Decoder).cuda()
    generator.load_state_dict(torch.load(checkpoint_name)['state_dict'])
    _ = generator.eval()

    checkpoint_name = model_path + opt.First_stage_model['checkpoint_encoder'] + '.pth'
    encoder = Encoder(dic=config.Encoder).cuda()
    encoder.load_state_dict(torch.load(checkpoint_name)['state_dict'])
    _ = encoder.eval()

    ### Create DDIM
    # control_dim = 0 if not opt.Training['control'] else opt.Training['control_dim']
    denoise_model = None
    network = DDIM.GaussianDiffusion(in_channels=opt.DDIM["in_channels"],
                                     condition_dim=opt.Conditioning_Model['z_dim'],
                                     denoise_model=denoise_model,
                                     image_size=opt.DDIM['image_size'],
                                     model_channels=opt.DDIM['model_channels'],
                                     out_channels=opt.DDIM['out_channels'],
                                     num_res_blocks=opt.DDIM['num_res_blocks'],
                                     attn_resolutions=opt.DDIM['attn_resolutions'],
                                     timesteps=opt.DDIM['timesteps'],
                                     loss_type=opt.DDIM['loss_type'],
                                     betas=opt.DDIM['beta'],
                                     dic=opt.Conditioning_Model,
                                     control=opt.Training['control']).cuda()

    ## Load Pytorch I3D for logging
    I3D     = FVD_logging.load_model().cuda() if config.Training['FVD']=='FVD' else DTFVD_Score.load_model(16).cuda()

    ###### Define Optimizer
    loss_func = loss.DiffusionLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=opt.Training['lr'], betas=(opt.Training['beta1'], opt.Training['beta2']),
                                 weight_decay=opt.Training['weight_decay'], amsgrad=opt.Training['amsgrad'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.Training['step_size'],
                                                gamma=opt.Training['gamma'])

    """==================== Dataloader ========================"""
    dloader = get_loader(opt.Data['dataset'], control=opt.Training['control'])
    train_dataset = dloader.Dataset(opt, mode='train')
    train_data_loader = torch.utils.data.DataLoader(train_dataset, num_workers=opt.Training['workers'],
                                                    batch_size=opt.Training['bs'], shuffle=True)
    eval_dataset = dloader.Dataset(opt, mode='eval')
    eval_data_loader = torch.utils.data.DataLoader(eval_dataset, num_workers=opt.Training['workers'],
                                                   batch_size=opt.Training['bs_eval'], shuffle=True)

    """======================Set Logging Files======================"""
    dt = datetime.now()
    dt = '{}-{}-{}-{}-{}-{}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    run_name = 'Stage2_' + opt.Data['dataset'] + '_Date-' + dt + '_' + opt.Training['savename']

    save_path = opt.Training['save_path'] + "/" + run_name

    ## Make the saving directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    opt.Training['save_path'] = save_path

    def make_folder(name):
        if not os.path.exists(name):
            os.makedirs(name)

    # Make summary plots, images, segmentation and videos folder
    make_folder(save_path + '/videos')

    # save yaml config
    OmegaConf.save(config=opt, f=save_path + "/config_stage2.yaml")

    # Create wandb logger
    log_dic = opt.Logging
    wandb.init(entity=log_dic['entitiy'], config=opt, dir=save_path, project=log_dic['project'],
               name=opt.Training['savename'], mode=log_dic['mode'])
    # wandb.watch(model, log="all")

    ## Offline logging
    logging_keys = ["Loss", "L2_Loss", 'PFVD']

    loss_track_train = aux.Logging(logging_keys[:-1])
    loss_track_test = aux.Logging(logging_keys[:-1])

    ### Setting up CSV writers
    full_log_train = aux.CSVlogger(save_path + "/log_per_epoch_train.csv", ["Epoch", "Time", "LR"] + logging_keys)
    full_log_eval = aux.CSVlogger(save_path + "/log_per_epoch_eval.csv", ["Epoch", "Time", "LR"] + logging_keys)

    """=================== Start training ! ==========================="""
    epoch_iterator = tqdm(range(0, opt.Training['n_epochs']), ascii=True, position=1)
    best_PFVD = 999

    for epoch in epoch_iterator:
        epoch_time = time.time()
        lr = [group['lr'] for group in optimizer.param_groups][0]

        ### Training ########
        epoch_iterator.set_description("Training with lr={}".format(np.round(lr, 6)))
        trainer(network, encoder, epoch, train_data_loader, loss_track_train, optimizer, loss_func, opt)

        ###### Validation #########
        epoch_iterator.set_description('Validating...')
        validator(network, encoder, epoch, eval_data_loader, loss_track_test, loss_func, opt)

        # ### Evaluation of PFVD score + log video samples
        epoch_iterator.set_description('Evaluation of FVD score ...')
        z_dim = config.Decoder['z_dim']
        z_h = config.Decoder['z_h']
        z_w = config.Decoder['z_w']
        if z_h is not None and z_w is not None:
            z_dim = z_dim * z_h * z_w
        PFVD = aux.evaluate_FVD_prior(eval_data_loader, network, generator, I3D, z_dim, opt, epoch,
                                      config.Training['FVD'], opt.Training['control'])
        wandb.log({'FVD': PFVD})

        save_dict = aux.get_save_dict(network, optimizer, scheduler, epoch)

        ## Save checkpoints
        if PFVD < best_PFVD:
            torch.save(save_dict, save_path + '/checkpoint_best_val.pth')
            best_PFVD = PFVD

        ###### Logging Epoch Data
        epoch_time = time.time() - epoch_time
        full_log_train.write([epoch, epoch_time, lr, *loss_track_train.log(), PFVD])
        full_log_eval.write([epoch, epoch_time, lr, *loss_track_test.log(), PFVD])

        ## Perform scheduler step
        scheduler.step()


### Start Training ###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--config", type=str, required=True, help="Define config file")
    parser.add_argument("-gpu", type=str, required=True)
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    aux.set_seed(42)
    main(conf)
