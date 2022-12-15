import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import copy
import fht

def Stat_dyn(dataloader, root, device, save = False):
    """ 
        Computes the variance of a predictor based on the identity
    """

    (inputs, classes) = next(iter(dataloader["train"]))
    inputs = inputs.cpu().detach().numpy();
    (batch_size, num_frames, channels, nx, ny) = inputs.shape;
    tot_num = 0;
    c = 0;
    Cov_had = np.zeros((nx*ny, nx*ny));
    with torch.no_grad():
        for inputs,labels in dataloader["train"]:
            c +=1;
            print("batch {}/{}".format(c, len(dataloader["train"])))
            inputs = inputs.to(device);
            (batch_size, num_frames,channels, nx, ny) = inputs.shape;
            tot_num +=batch_size*(num_frames-1);
            output = inputs[:,1:, :,:,:];
            inputs = inputs[:,:-1,:,:,:]
            images = output-inputs;
            images = images.view(batch_size*(num_frames-1)*channels,1, nx, ny);
            images = images.cpu().detach().numpy();
            for i in range(images.shape[0]):
                img = images[i,0,:,:];
                Norm_Variable = np.reshape(img, (nx*ny,1));
                Cov_had += Norm_Variable*np.transpose(Norm_Variable);
        Cov_had = Cov_had/(tot_num-1);
    if save :
        np.save(root+'Cov_dyn_{}x{}'.format(nx,ny)+'.npy', Cov_had)
    return Cov_had 


