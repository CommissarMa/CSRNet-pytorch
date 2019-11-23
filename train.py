#%%
import numpy as np
import time
import torch
import torch.nn as nn
import os
import sys
from tqdm import tqdm

from config import Config
from model import CSRNet
from dataset import create_train_dataloader,create_test_dataloader
from utils import denormalize

if __name__=="__main__":
    
    cfg = Config()                                                          # configuration
    model = CSRNet().to(cfg.device)                                         # model
    criterion = nn.MSELoss(size_average=False)                              # objective
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.lr)              # optimizer
    train_dataloader = create_train_dataloader(cfg.dataset_root, use_flip=True, batch_size=cfg.batch_size)
    test_dataloader  = create_test_dataloader(cfg.dataset_root)             # dataloader

    min_mae = sys.maxsize
    min_mae_epoch = -1
    for epoch in range(1, cfg.epochs):                          # start training
        model.train()
        epoch_loss = 0.0
        for i, data in enumerate(tqdm(train_dataloader)):
            image = data['image'].to(cfg.device)
            gt_densitymap = data['densitymap'].to(cfg.device)
            et_densitymap = model(image)                        # forward propagation
            loss = criterion(et_densitymap,gt_densitymap)       # calculate loss
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()                                     # back propagation
            optimizer.step()                                    # update network parameters
        cfg.writer.add_scalar('Train_Loss', epoch_loss/len(train_dataloader), epoch)

        model.eval()
        with torch.no_grad():
            epoch_mae = 0.0
            for i, data in enumerate(tqdm(test_dataloader)):
                image = data['image'].to(cfg.device)
                gt_densitymap = data['densitymap'].to(cfg.device)
                et_densitymap = model(image).detach()           # forward propagation
                mae = abs(et_densitymap.data.sum()-gt_densitymap.data.sum())
                epoch_mae += mae.item()
            epoch_mae /= len(test_dataloader)
            if epoch_mae < min_mae:
                min_mae, min_mae_epoch = epoch_mae, epoch
                torch.save(model.state_dict(), os.path.join(cfg.checkpoints,str(epoch)+".pth"))     # save checkpoints
            print('Epoch ', epoch, ' MAE: ', epoch_mae, ' Min MAE: ', min_mae, ' Min Epoch: ', min_mae_epoch)   # print information
            cfg.writer.add_scalar('Val_MAE', epoch_mae, epoch)
            cfg.writer.add_image(str(epoch)+'/Image', denormalize(image[0].cpu()))
            cfg.writer.add_image(str(epoch)+'/Estimate density count:'+ str('%.2f'%(et_densitymap[0].cpu().sum())), et_densitymap[0]/torch.max(et_densitymap[0]))
            cfg.writer.add_image(str(epoch)+'/Ground Truth count:'+ str('%.2f'%(gt_densitymap[0].cpu().sum())), gt_densitymap[0]/torch.max(gt_densitymap[0]))
            
# %%
