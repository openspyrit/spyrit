# In a terminal
# python train.py --CR 1024 --precompute_root ./stats_walsh/ --num_epochs 30 --batch_size 512

# In spyder :
# runfile('train_noisy.py', args='--CR 1024 --precompute_root ./stats_walsh/ --num_epochs 30 --batch_size 512 --intensity_max 50')
#%%
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import argparse
import sys
from pathlib import Path
import scipy.io as sio
from spyrit.learning.model_Had_DCAN import *
from spyrit.learning.nets import *
from spyrit.misc.disp import *
import spyrit.misc.walsh_hadamard as wh

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Acquisition
    parser.add_argument("--img_size",   type=int,   default=64,     help="Height / width dimension")
    parser.add_argument("--CR",         type=int,   default=512,    help="Number of patterns")
    # Network and training
    parser.add_argument("--data_root",  type=str,   default='./data/', help="Path to SLT-10 dataset")
    parser.add_argument("--net_arch",   type=int,   default=0,      help="Network architecture (variants for the FCL)")
    parser.add_argument("--precompute_root", type=str, default='./models/', help="Path to precomputed data")
    parser.add_argument("--precompute", default=False, action='store_true', help="recompute, even if precomputed data available")
    parser.add_argument("--model_root", type=str,   default='./models/', help="Path to model saving files")
    parser.add_argument("--intensity_max",  type=float,   default=2500, help="maximum photons/pixel")
    parser.add_argument("--intensity_sig",  type=float,   default=0.5, help="std of maximun photons/pixel")
    parser.add_argument("--no_denoi",   default=False, action='store_true', help="No denoising layer")
    # Optimisation
    parser.add_argument("--num_epochs", type=int,   default=20,     help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=256,   help="Size of each training batch")
    parser.add_argument("--reg",        type=float, default=1e-7,   help="Regularisation Parameter")
    parser.add_argument("--lr",         type=float, default=1e-3,   help="Learning Rate")
    parser.add_argument("--step_size",  type=int,   default=10,     help="Scheduler Step Size")
    parser.add_argument("--gamma",      type=float, default=0.5,    help="Scheduler Decrease Rate")
    parser.add_argument("--checkpoint_model", type=str, default="", help="Optional path to checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=0, help="Interval between saving model checkpoints"
    )
    opt = parser.parse_args()
    opt.precompute_root = Path(opt.precompute_root)
    opt.model_root = Path(opt.model_root)
    print(opt)

    #%% =======================================================================
    # 0. Setting up parameters for training
    # =========================================================================
    # The device of the machine, number of workers...
    #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    #%% =======================================================================
    # 1. Load and normalize STL10
    # =========================================================================
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1]. Also
    # RGB images transformed into grayscale images.
    transform = transforms.Compose(
        [transforms.functional.to_grayscale,
         transforms.Resize((opt.img_size, opt.img_size)),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    
    trainset = \
        torchvision.datasets.STL10(root=opt.data_root, split='train+unlabeled',download=True, transform=transform)
    trainloader = \
        torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,shuffle=False)

    testset = \
        torchvision.datasets.STL10(root=opt.data_root, split='test',download=True, transform=transform)
    testloader = \
        torch.utils.data.DataLoader(testset, batch_size=opt.batch_size,shuffle=False)

    dataloaders = {'train':trainloader, 'val':testloader}

    #%% =======================================================================
    # 2. Compute mean and covariance of the training images
    # =========================================================================
    my_average_file = Path(opt.precompute_root) / ('Average_{}x{}'.format(opt.img_size, opt.img_size)+'.npy')
    my_cov_file = Path(opt.precompute_root) / ('Cov_{}x{}'.format(opt.img_size, opt.img_size)+'.npy')

    Path(opt.precompute_root).mkdir(parents=True, exist_ok=True)
    if not(my_average_file.is_file()) or not(my_cov_file.is_file()) or opt.precompute:
        print('Computing covariance and mean (overwrite previous files)')
        Mean_had, Cov_had = Stat_had(trainloader, opt.precompute_root)
    else:
        print('Loading covariance and mean')
        Mean_had = np.load(my_average_file)
        Cov_had  = np.load(my_cov_file)
    
    Ord = Cov2Var(Cov_had)
    H =  wh.walsh2_matrix(opt.img_size)/opt.img_size
    Cov_had /= opt.img_size**2
    #%% =======================================================================
    # 3. Define a Neural Network
    # =========================================================================
    net_type = ['c0mp', 'comp','pinv', 'free']
    if opt.intensity_max==float('inf'):
        model = compNet(opt.img_size, opt.CR, Mean_had, Cov_had, 
                    variant=opt.net_arch, H=H, Ord=Ord)
        midfix = ''
    elif opt.no_denoi==True:
        model = noiCompNet(opt.img_size, opt.CR, Mean_had, Cov_had, 
                variant=opt.net_arch, N0 = opt.intensity_max, 
                sig = opt.intensity_sig,  H=H, Ord=Ord)
        midfix = '_N0_{}_sig_{}'.format(opt.intensity_max, opt.intensity_sig)
    else:
        model = DenoiCompNet(opt.img_size, opt.CR, Mean_had, Cov_had, 
                variant=opt.net_arch, N0 = opt.intensity_max, 
                sig = opt.intensity_sig,  H=H, Ord=Ord)
        midfix = '_N0_{}_sig_{}_Denoi'.format(opt.intensity_max, opt.intensity_sig)
        
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    if opt.checkpoint_model:
        model.load_state_dict(torch.load(opt.checkpoint_model))

    #%% =======================================================================
    # 4. Define a Loss function optimizer and scheduler
    # =========================================================================
    # Penalization defined in DCAN.py
    loss = nn.MSELoss();
    criterion = Weight_Decay_Loss(loss);
    optimizer = optim.Adam(model.parameters(), lr=opt.lr);
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)

    #%% =======================================================================
    # 5. Train the network
    # =========================================================================
    #- training parameters
    suffix = '_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
           opt.img_size, opt.CR, opt.num_epochs, opt.lr, opt.step_size,\
           opt.gamma, opt.batch_size, opt.reg)
    title = opt.model_root / ('NET_'+net_type[opt.net_arch]+midfix+suffix)
    print(title)
    Path(opt.model_root).mkdir(parents=True, exist_ok=True)
    
    if opt.checkpoint_interval:
        Path(title).mkdir(parents=True, exist_ok=True)
    #
    model, train_info = train_model(model, criterion, \
            optimizer, scheduler, dataloaders, device, title, num_epochs=opt.num_epochs,\
            disp=True, do_checkpoint=opt.checkpoint_interval)

    #%% =======================================================================
    # 6. Saving the model
    # =========================================================================
    #- network's architecture
    save_net(title, model)

    #- save training history
    params = Train_par(opt.batch_size, opt.lr, opt.img_size,reg=opt.reg);
    params.set_loss(train_info);
    train_path = opt.model_root / ('TRAIN_'+net_type[opt.net_arch]+midfix+suffix+'.pkl')
    with open(train_path, 'wb') as param_file:
        pickle.dump(params,param_file)
