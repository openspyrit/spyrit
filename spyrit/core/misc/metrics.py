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

def batch_psnr(torch_batch, output_batch):
    list_psnr = [];
    for i in range(torch_batch.shape[0]):
        img = torch_batch[i, 0, :, :];
        img_out = output_batch[i,0,:,:];
        img = img.cpu().detach().numpy();
        img_out = img_out.cpu().detach().numpy();
        list_psnr.append(psnr(img, img_out));
    return list_psnr;

def batch_ssim(torch_batch, output_batch):
    list_ssim = [];
    for i in range(torch_batch.shape[0]):
        img = torch_batch[i, 0, :, :];
        img_out = output_batch[i,0,:,:];
        img = img.cpu().detach().numpy();
        img_out = img_out.cpu().detach().numpy();
        list_ssim.append(ssim(img, img_out));
    return list_ssim;

def dataset_meas(dataloader, model, device):
    meas = [];
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        # with torch.no_grad():
        b,c,h,w = inputs.shape;
        net_output = model.acquire(inputs, b, c, h, w);
        raw = net_output[:, 0, :];
        raw = raw.cpu().detach().numpy();
        meas.extend(raw);          
    return meas 
#
#def dataset_psnr_different_measures(dataloader, model, model_2, device):
#    psnr = [];
#    #psnr_fc = [];   
#    for inputs, labels in dataloader:
#        inputs = inputs.to(device)
#        m = model_2.normalized measure(inputs);
#        net_output  = model.forward_reconstruct(inputs);
#        #net_output2 = model.evaluate_fcl(inputs);
#    
#        psnr += batch_psnr(inputs, net_output);
#        #psnr_fc += batch_psnr(inputs, net_output2);               
#    psnr = np.array(psnr);
#    #psnr_fc = np.array(psnr_fc);
#    return psnr;
#

def dataset_psnr(dataloader, model, device):
    psnr = [];
    psnr_fc = [];   
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        #with torch.no_grad():
        #b,c,h,w = inputs.shape;
        
        net_output  = model.evaluate(inputs);
        net_output2 = model.evaluate_fcl(inputs);
    
        psnr += batch_psnr(inputs, net_output);
        psnr_fc += batch_psnr(inputs, net_output2);               
    psnr = np.array(psnr);
    psnr_fc = np.array(psnr_fc);
    return psnr, psnr_fc

def dataset_ssim(dataloader, model, device):
    ssim = [];
    ssim_fc = [];   
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        # evaluate full model and fully connected layer
        net_output  = model.evaluate(inputs);
        net_output2 = model.evaluate_fcl(inputs);
        # compute SSIM and concatenate
        ssim += batch_ssim(inputs, net_output);
        ssim_fc += batch_ssim(inputs, net_output2);               
    ssim = np.array(ssim);
    ssim_fc = np.array(ssim_fc);
    return ssim, ssim_fc

def dataset_psnr_ssim(dataloader, model, device):
    # init lists
    psnr = [];
    ssim = [];  
    # loop over batches
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        # evaluate full model
        net_output  = model.evaluate(inputs);
        # compute PSNRs and concatenate
        psnr += batch_psnr(inputs, net_output);
        # compute SSIMs and concatenate
        ssim += batch_ssim(inputs, net_output); 
    # convert
    psnr = np.array(psnr);           
    ssim = np.array(ssim);
    return psnr, ssim

def dataset_psnr_ssim_fcl(dataloader, model, device):
    # init lists
    psnr = []; 
    ssim = [];   
    # loop over batches
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        # evaluate fully connected layer
        net_output  = model.evaluate_fcl(inputs);
        # compute PSNRs and concatenate
        psnr += batch_psnr(inputs, net_output);
        # compute SSIMs and concatenate
        ssim += batch_ssim(inputs, net_output);   
    # convert
    psnr = np.array(psnr);            
    ssim = np.array(ssim);
    return psnr, ssim

def psnr(I1,I2):
    """
    Computes the psnr between two images I1 and I2
    """
    d=np.amax(I1)-np.amin(I1);
    diff=np.square(I2-I1);
    MSE=diff.sum()/I1.size;
    Psnr=10*np.log(d**2/MSE)/np.log(10);
    return Psnr

def psnr_(img1,img2,r=2):
    """
    Computes the psnr between two image with values expected in a given range
    
    Args:
        img1, img2 (np.ndarray): images
        r (float): image range
        
    Returns:
        Psnr (float): Peak signal-to-noise ratio
    
    """
    MSE = np.mean((img1 - img2) ** 2)
    Psnr = 10*np.log(r**2/MSE)/np.log(10);
    return Psnr
    
def ssim(I1,I2):
    """
    Computes the ssim between two images I1 and I2
    """
    L = np.amax(I1)-np.amin(I1);
    mu1 = np.mean(I1);
    mu2 = np.mean(I2);
    s1 = np.std(I1);
    s2 = np.std(I2);
    s12 = np.mean(np.multiply((I1-mu1), (I2-mu2)));
    c1 = (0.01*L)**2
    c2 = (0.03*L)**2
    result = ((2*mu1*mu2+c1)*(2*s12+c2))/((mu1**2+mu2**2+c1)*(s1**2+s2**2+c2))
    return result

def batch_psnr_vid(input_batch, output_batch):
    list_psnr = [];
    batch_size, seq_length, c, h, w = input_batch.shape
    input_batch = input_batch.view(batch_size * seq_length*c, 1, h, w) 
    output_batch = output_batch.view(batch_size * seq_length*c, 1, h, w) 
    for i in range(input_batch.shape[0]):
        img = input_batch[i, 0, :, :];
        img_out = output_batch[i,0,:,:];
        img = img.cpu().detach().numpy();
        img_out = img_out.cpu().detach().numpy();
        list_psnr.append(psnr(img, img_out));
    return list_psnr;


def batch_ssim_vid(input_batch, output_batch):
    list_ssim = [];
    batch_size, seq_length, c, h, w = input_batch.shape
    input_batch = input_batch.view(batch_size * seq_length*c, 1, h, w) 
    output_batch = output_batch.view(batch_size * seq_length*c, 1, h, w) 
    for i in range(input_batch.shape[0]):
        img = input_batch[i, 0, :, :];
        img_out = output_batch[i,0,:,:];
        img = img.cpu().detach().numpy();
        img_out = img_out.cpu().detach().numpy();
        list_ssim.append(ssim(img, img_out));
    return list_ssim;


def compare_video_nets_supervised(net_list,testloader, device):
    psnr = [[] for i in range(len(net_list))];
    ssim = [[] for i in range(len(net_list))];
    for batch, (inputs, labels) in enumerate(testloader):
        [batch_size, seq_length, c, h, w] = inputs.shape;
        print("Batch :{}/{}".format(batch+1, len(testloader)));
        inputs = inputs.to(device);
        labels = labels.to(device);
        with torch.no_grad():
            for i in range(len(net_list)):
               outputs = net_list[i].evaluate(inputs);
               psnr[i]+= batch_psnr_vid(labels, outputs);
               ssim[i]+= batch_ssim_vid(labels, outputs);
    return psnr, ssim

def compare_nets_unsupervised(net_list ,testloader, device):
    psnr = [[] for i in range(len(net_list))];
    ssim = [[] for i in range(len(net_list))];
    for batch, (inputs, labels) in enumerate(testloader):
        [batch_size, seq_length, c, h, w] = inputs.shape;
        print("Batch :{}/{}".format(batch+1, len(testloader)));
        inputs = inputs.to(device);
        labels = labels.to(device);
        with torch.no_grad():
            for i in range(len(net_list)):
               outputs = net_list[i].evaluate(inputs);
               psnr[i]+= batch_psnr_vid(outputs, labels);
               ssim[i]+= batch_ssim_vid(outputs, labels);
    return psnr, ssim

def print_mean_std(x, tag=''):  
    print("{}psnr = {} +/- {}".format(tag,np.mean(x),np.std(x)))
