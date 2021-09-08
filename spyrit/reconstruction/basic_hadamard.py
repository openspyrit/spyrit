# -----------------------------------------------------------------------------
#   This software is distributed under the terms
#   of the GNU Lesser General  Public Licence (LGPL)
#   See LICENSE.md for further details
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import imageio
import cv2
import matplotlib.pyplot as plt
import PIL



def optimized_order(input_batch, mask):
    output_batch = torch.zeros(input_batch.shape);
    for i in range(input_batch.shape[0]):
        img = input_batch[i, 0, :, :];
        #img = uint8(img.cpu().detach().numpy());
        img = img.cpu().detach().numpy();
        
        img= img.astype('float64')

        H = walsh_matrix(len(img))
        img_had = wh.walsh2(img,H)/len(img)
        img_had = np.multiply(img_had, msk);
        H = walsh_matrix(len(img_had))
        img_out = wh.walsh2(img_had,H)/len(img_had)

        output_batch[i,0,:,:] = torch.from_numpy(img_out);
    return output_batch




def optimal_order_mask(CR, title1, trainloader, img_size):
    Cumulated_had = optim_had(trainloader,title1);
    msk = np.ones((img_size,img_size))
    had_comp = np.reshape(rankdata(-Cumulated_had, method = 'ordinal'),(img_size, img_size));
    msk[np.absolute(had_comp)>CR]=0;
    return msk;

def optimal_order_noisy(net, input_batch, dyn, No, fc):
    batch_size,c,h,w = x.shape
    Patt = list(net.children())[0];
    P, T = split(Patt, dyn);
    M = net.M;
    
    x = input_batch;
    x = x.view(batch_size*c, 1, h, w);
    x = (x+1)/2;
    x = No*x;
    x = P(x);
    x = x.view(batch_size*c, 1, 2*M);
    noise = torch.distributions.poisson.Poisson(f);
    x = noise.sample();
    x = T(x);
    x = 2*x - torch.reshape(Patt(torch.ones((batch_size*c, 1, h, w))),(batch_size*c, 1, M));
    x = fc(x);
    return x;

def stat_completion(x, net):
    M = model.M;
    batch_size,c,h,w = x.shape
    x = x.view(batch_size*c, 1, h, w);
    x = (x+1)/2;
    x = No*x;
    x = model.x
    x = x.view([x.shape[0],1,M])
    x = model.fc1(x)
    output_batch = x.view([x.shape[0], 1, h, w])
    return output_batch;

def stat_completion_noisy(net, input_batch, No):
    batch_size,c,h,w = x.shape
    fc1 = net.fc1;
    Patt = net.Patt;
    P , T = net.P, net.T;
    M = net.M;
    
    x = input_batch;

    x = x.view(batch_size*c, 1, h, w);
    x = (x+1)/2;
    x = No*x;
    x = P(x);
    x = x.view(batch_size*c, 1, 2*M);
    noise = torch.distributions.poisson.Poisson(f);
    x = noise.sample();
    x = T(x);
    x =(1/No)*(2*x - torch.reshape(Patt(torch.ones((batch_size*c, 1, h, w))),(batch_size*c, 1, M)));
    x = fc1(x);
    return x;

def noisy_net_output(net, input_batch, No):
    batch_size,c,h,w = x.shape;
    Patt = net.Patt;
    P , T = net.P, net.T;
    M = net.M;
    
    x = input_batch;
    

    x = x.view(batch_size*c, 1, h, w);
    x = (x+1)/2;
    x = No*x;
    x = P(x);
    x = x.view(batch_size*c, 1, 2*M);
    noise = torch.distributions.poisson.Poisson(f);
    x = noise.sample();
    x = net.recon(x);
    return x;

def rescale(model, n):
    model.conv1.weight.data = n*model.conv1.weight.data;
    model.fc1.weight.data = (1/n)*model.fc1.weight.data;
    model.fc1.bias.data = (1/n)*model.fc1.bias.data;


def inverse_transform_net(Cov_had, Mean_had, M):
    n, _ = Mean_had.shape;
    _, _, _, P, H = Hadamard_stat_completion_matrices(Cov_had, Mean_had, M);
    H = n*H;
    S = np.zeros((M,n**2));
    S[:,:M]=np.eye(M);
    Q = (1/n)**2 * np.dot(np.dot(np.transpose(H), np.transpose(P)), np.transpose(S));
    fc = nn.Linear(M,n**2, bias = False);

    fc.weight.data=torch.from_numpy(Q);
    fc.weight.data=fc.weight.data.float();
    fc.bias.requires_grad = False;
    fc.weight.requires_grad=False;

    return fc;

def optimized_order_vid(input_batch, msk):
    batch_size, seq_length, c, h, w = input_batch.shape
    input_batch = input_batch.view(batch_size * seq_length*c, 1, h, w) 
    output_batch = torch.zeros(input_batch.shape);
    for i in range(input_batch.shape[0]):
        img = input_batch[i, 0, :, :];
        img = img.cpu().detach().numpy();
        img= img.astype('float64')
        H = walsh_matrix(len(img))
        img_had = wh.walsh2(img,H)/len(img)
        img_had = np.multiply(img_had, msk);
        H = walsh_matrix(len(img_had))
        img_out = wh.walsh2(img_had,H)/len(img_had)

        output_batch[i,0,:,:] = torch.from_numpy(img_out);
    output_batch = output_batch.view([batch_size, seq_length, c, h, w])
    return output_batch


