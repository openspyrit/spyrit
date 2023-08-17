# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:25:43 2022

@author: ducros
"""

from __future__ import print_function, division
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.profiler
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import argparse
from pathlib import Path
import pickle
import os
from datetime import datetime
import math

from spyrit.misc.walsh_hadamard import walsh2_matrix
from spyrit.misc.sampling import Permutation_Matrix

from spyrit.core.recon import DCNet, PinvNet, UPGD
from spyrit.core.train import train_model, Train_par, save_net, Weight_Decay_Loss
from spyrit.core.nnet import Unet, ConvNet, ConvNetBN
from spyrit.misc.statistics import Cov2Var, data_loaders_ImageNet, data_loaders_stl10

# pip install -e git+https://github.com/openspyrit/spas.git@v1.4#egg=spas
# python3 ./spyrit-examples/2022_OE_spyrit2/download_data.py

# python tuto_train_pseudoinverse_cnn_linear.py --data_root ../../data/ --model_root ./model/ --stat_root ./stat/ --tb_path ./runs/runs_stdl10_n100_m1024/ --data stl10 --N0 100 --M 1024 --num_epochs 30 --batch_size 512 --lr 1e-3 --step_size 10 --gamma 0.5 --reg 1e-7 --arch dc-net --denoi unet --device cuda:0
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Acquisition
    parser.add_argument("--img_size",   type=int,   default=64,   help="Height / width dimension")
    parser.add_argument("--M",          type=int,   default=512,  help="Number of patterns")
    parser.add_argument("--subs",       type=str,   default="var",  help="Among 'var','rect'")
    
    # Network and training
    parser.add_argument("--data",       type=str,   default="stl10", help="stl10 or imagenet")
    parser.add_argument("--model_root", type=str,   default='./model/', help="Path to model saving files")
    parser.add_argument("--data_root",  type=str,   default="./data/", help="Path to the dataset")
    
    parser.add_argument("--N0",         type=float, default=10,   help="Mean maximum total number of photons")
    parser.add_argument("--stat_root",  type=str,   default="./stat/", help="Path to precomputed data")
    parser.add_argument("--arch",       type=str,   default="dc-net", help="Choose among 'dc-net','pinv-net',")
    parser.add_argument("--denoi",      type=str,   default="unet", help="Choose among 'cnn','cnnbn', 'unet'")
    parser.add_argument("--device",     type=str,   default="", help="Choose among 'cuda','cpu'")
    #parser.add_argument("--no_denoi",   default=False, action='store_true', help="No denoising layer")


    # Optimisation
    parser.add_argument("--num_epochs", type=int,   default=30,   help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=512, help="Size of each training batch")
    parser.add_argument("--reg",        type=float, default=1e-7, help="Regularisation Parameter")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--step_size",  type=int,   default=10,   help="Scheduler Step Size")
    parser.add_argument("--gamma",      type=float, default=0.5,  help="Scheduler Decrease Rate")
    parser.add_argument("--checkpoint_model", type=str, default="", help="Optional path to checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=0, help="Interval between saving model checkpoints")
    
    # Tensorboard
    parser.add_argument("--tb_path",    type=str,   default=False, help="Relative path for Tensorboard experiment tracking logs")
    parser.add_argument("--tb_prof",    type=bool,   default=False, help="Profiler for code with Tensorboard")

    opt = parser.parse_args()
    opt.model_root = Path(opt.model_root)
    opt.data_root = Path(opt.data_root)
    
    # Define parameters
    opt.data_root = Path("../data/")
    opt.subs = 'had+'
    opt.M = 1024
    opt.arch = 'pinv-net'
    opt.denoi = 'cnn'
    opt.tb_prof = True
    now = datetime.now().strftime('%Y-%m-%d_%H-%M')
    opt.tb_path = f'runs/runs_stdl10_n1_m1024/{now}'
    opt.num_epochs = 1
    opt.N0 = 1

    print(opt)
    
    #==========================================================================
    # 0. Setting up parameters for training
    #==========================================================================
    # The device of the machine, number of workers...
    # 
    if opt.device: 
        device = torch.device(opt.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    
    #==========================================================================
    # 1. Loading and normalizing data
    #==========================================================================
    if opt.data == 'stl10':
        dataloaders = data_loaders_stl10(opt.data_root, 
                                        img_size=opt.img_size, 
                                        batch_size=opt.batch_size, 
                                        seed=7,
                                        shuffle=True, download=True)  

        

    elif opt.data == 'imagenet':
        dataloaders = data_loaders_ImageNet(opt.data_root / 'test', 
                                        opt.data_root / 'val', 
                                        img_size=opt.img_size, 
                                        batch_size=opt.batch_size, 
                                        seed=7,
                                        shuffle=True)
   
    #==========================================================================
    # 2. Subsampling
    #==========================================================================
    print('Subsampling: low frequency (rect)')      
    h = opt.img_size
    M = opt.M
    F = walsh2_matrix(h)
    F = np.where(F>0, F, 0)
    Sampling_map = np.ones((h,h))
    M_xy = math.ceil(M**0.5)
    Sampling_map[:,M_xy:] = 0
    Sampling_map[M_xy:,:] = 0
    Perm = Permutation_Matrix(Sampling_map)
    F = Perm@F 
    H = F[:M,:]

    #==========================================================================
    # 3. Define Measuremt operators
    #==========================================================================
    from spyrit.core.meas import Linear
    from spyrit.core.noise import NoNoise
    from spyrit.core.prep import DirectPoisson

    meas_op = Linear(H, pinv=True)  

    # Noiseless case
    noise = NoNoise(meas_op)        
    prep = DirectPoisson(1.0, meas_op) # "Undo" the NoNoise operator

    #meas = HadamSplit(opt.M, opt.img_size, Ord)
    #prep = SplitPoisson(opt.N0, meas)
    #noise = PoissonApproxGauss(meas, opt.N0) # faster than Poisson
    
    # Image-domain denoising layer
    if opt.denoi == 'cnn':      # CNN no batch normalization
        denoi = ConvNet()
    elif opt.denoi == 'cnnbn':  # CNN with batch normalization
        denoi = ConvNetBN()
    elif opt.denoi == 'unet':   # Unet
        denoi = Unet()    
        
    if opt.arch == 'pinv-net':    # Pseudo Inverse Network
        model = PinvNet(noise, prep, denoi)
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    if opt.checkpoint_model:
        model.load_state_dict(torch.load(opt.checkpoint_model))

    #==========================================================================
    # 4. Define a Loss function optimizer and scheduler
    #==========================================================================
    # Penalization defined in DCAN.py
    loss = nn.MSELoss();
    criterion = Weight_Decay_Loss(loss);
    optimizer = optim.Adam(model.parameters(), lr=opt.lr);
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)
    
    #==========================================================================
    # 5. Train the network
    #==========================================================================
    # We  loop over our data iterator, feed the inputs to the
    model, train_info = train_model(model, criterion, \
            optimizer, scheduler, dataloaders, device, opt.model_root, num_epochs=opt.num_epochs,\
            disp=True, do_checkpoint=opt.checkpoint_interval, tb_path=opt.tb_path)
    
    #==========================================================================
    # 6. Saving the model so that it can later be utilized
    #==========================================================================
    #- network's architecture
    train_type = 'N0_{:g}'.format(opt.N0) 
        
    #- training parameters
    suffix = 'N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
           opt.img_size, opt.M, opt.num_epochs, opt.lr, opt.step_size,\
           opt.gamma, opt.batch_size, opt.reg)

    title = opt.model_root / f'{opt.arch}_{opt.denoi}_{opt.data}_{train_type}_{suffix}'    
    print(title)
    
    Path(opt.model_root).mkdir(parents=True, exist_ok=True)
   
    if opt.checkpoint_interval:
       Path(title).mkdir(parents=True, exist_ok=True)
       
    save_net(title, model)
    
    #- save training history
    params = Train_par(opt.batch_size, opt.lr, opt.img_size,reg=opt.reg);
    params.set_loss(train_info);
    train_path = opt.model_root / f'TRAIN_{opt.arch}_{opt.denoi}_{opt.data}_{train_type}_{suffix}.pkl'
    
    with open(train_path, 'wb') as param_file:
        pickle.dump(params,param_file)
    torch.cuda.empty_cache()

