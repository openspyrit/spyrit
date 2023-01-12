# ==================================================================================
#from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import poisson
from collections import OrderedDict
#from scipy.sparse.linalg import aslinearoperator
#from pylops_gpu import Diagonal, LinearOperator
#from pylops_gpu.optimization.cg import cg --- currently not working
#from pylops_gpu.optimization.leastsquares import NormalEquationsInversion
from spyrit.misc.walsh_hadamard import walsh2_torch, walsh_matrix
from typing import Union

# ==================================================================================
# Image Domain denoising layers
# ==================================================================================
# ===========================================================================================
class Unet(nn.Module):
# ===========================================================================================
    def __init__(self, in_channel=1, out_channel=1):
        super(Unet, self).__init__()
        #Descending branch
        self.conv_encode1 = self.contract(in_channels=in_channel, out_channels=16)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contract(16, 32)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contract(32, 64)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        #Bottleneck
        self.bottleneck = self.bottle_neck(64)
        #Decode branch
        self.conv_decode4 = self.expans(64,64,64)
        self.conv_decode3 = self.expans(128, 64, 32)
        self.conv_decode2 = self.expans(64, 32, 16)
        self.final_layer = self.final_block(32, 16, out_channel)
        
    
    def contract(self, in_channels, out_channels, kernel_size=3, padding=1):
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=padding),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=padding),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels),
                    )
        return block
    
    def expans(self, in_channels, mid_channel, out_channels, kernel_size=3,padding=1):
            block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=padding),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=padding),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=kernel_size, stride=2,padding=padding, output_padding=1)
                    )

            return block
    

    def concat(self, upsampled, bypass):
        out = torch.cat((upsampled,bypass),1)
        return out
    
    def bottle_neck(self,in_channels, kernel_size=3, padding=1):
        bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=2*in_channels, padding=padding),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=2*in_channels, out_channels=in_channels, padding=padding),
            torch.nn.ReLU(),
            )
        return bottleneck
    
    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
            block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel,padding=1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel,padding=1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
                    )
            return  block
    
    def forward(self,x):
        
        #Encode
        encode_block1 = self.conv_encode1(x)
        x = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(x)
        x = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(x)
        x = self.conv_maxpool3(encode_block3)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decode
        x = self.conv_decode4(x)
        x = self.concat(x, encode_block3)
        x = self.conv_decode3(x)
        x = self.concat(x, encode_block2)
        x = self.conv_decode2(x)
        x = self.concat(x, encode_block1)
        x = self.final_layer(x)      
        return x
    
# ===========================================================================================
class ConvNet(nn.Module):
# ===========================================================================================
    def __init__(self):
        super(ConvNet,self).__init__()
        self.convnet = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(1,64,kernel_size=9, stride=1, padding=4)),
                ('relu1', nn.ReLU()),
                ('conv2', nn.Conv2d(64,32,kernel_size=1, stride=1, padding=0)),
                ('relu2', nn.ReLU()),
                ('conv3', nn.Conv2d(32,1,kernel_size=5, stride=1, padding=2))
                ]));
                
    def forward(self,x):
        x = self.convnet(x)
        return x
    
# ===========================================================================================
class ConvNetBN(nn.Module):  
# ===========================================================================================
    def __init__(self):
        super(ConvNetBN,self).__init__()
        self.convnet = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(1,64,kernel_size=9, stride=1, padding=4)),
                ('relu1', nn.ReLU()),
                ('BN1', nn.BatchNorm2d(64)),
                ('conv2', nn.Conv2d(64,32,kernel_size=1, stride=1, padding=0)),
                ('relu2', nn.ReLU()),
                ('BN2', nn.BatchNorm2d(32)),
                ('conv3', nn.Conv2d(32,1,kernel_size=5, stride=1, padding=2))
                ]))
        
    def forward(self,x):
        x = self.convnet(x)
        return x
    
# ===========================================================================================
class DConvNet(nn.Module):  
# ===========================================================================================
    def __init__(self):
        super(DConvNet,self).__init__()
        self.convnet = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(1,64,kernel_size=9, stride=1, padding=4)),
                ('relu1', nn.ReLU()),
                ('BN1', nn.BatchNorm2d(64)),
                ('conv2', nn.Conv2d(64,64,kernel_size=1, stride=1, padding=0)),
                ('relu2', nn.ReLU()),
                ('BN2', nn.BatchNorm2d(64)),
                ('conv3', nn.Conv2d(64,32,kernel_size=3, stride=1, padding=1)),
                ('relu3', nn.ReLU()),
                ('BN3', nn.BatchNorm2d(32)),
                ('conv4', nn.Conv2d(32,1,kernel_size=5, stride=1, padding=2))
                ]));
    def forward(self,x):
        x = self.convnet(x)
        return x
    
# ===========================================================================================
class List_denoi(nn.Module):  
# ===========================================================================================
    def __init__(self, Denoi, n_denoi):
        super(List_denoi,self).__init__()
        self.n_denoi = n_denoi;
        conv_list = [];
        for i in range(n_denoi):
            conv_list.append(copy.deepcopy(Denoi));
        self.conv = nn.ModuleList(conv_list);
        
    def forward(self,x,iterate):
        index = min(iterate, self.n_denoi-1); 
        x = self.conv[index](x);
        return x


# ===========================================================================================
class Identity(nn.Module):  # Can be useful for ablation study
# ===========================================================================================
    def __init__(self):
        super(Identity,self).__init__()
        
    def forward(self,x):
        return x

# ==================================================================================
# Complete Reconstruction method
# ==================================================================================
# ===========================================================================================
class DC_Net(nn.Module):
# ===========================================================================================
    def __init__(self, Acq, PreP, DC_layer, Denoi):
        super().__init__()
        self.Acq = Acq  # must be a split operator for now
        self.PreP = PreP
        self.DC_layer = DC_layer
        self.Denoi = Denoi

    def forward(self, x):
        # x - of shape [b,c,h,w]
        b,c,h,w = x.shape

        # Acquisition
        x = x.view(b*c,h*w)    # shape x = [b*c,h*w] = [b*c,N]
        x_0 = torch.zeros_like(x).to(x.device)
        x = self.Acq(x) #  shape x = [b*c, 2*M]

        # Preprocessing
        sigma_noi = self.PreP.sigma(x)
        x = self.PreP(x, self.Acq.FO) # shape x = [b*c, M]

        # Data consistency layer
        # measurements to the image domain 
        x = self.DC_layer(x, x_0, sigma_noi, self.Acq.FO) # shape x = [b*c, N]

        # Image-to-image mapping via convolutional networks 
        # Image domain denoising 
        x = x.view(b*c,1,h,w)
        x = self.Denoi(x); # shape stays the same

        x = x.view(b,c,h,w)
        return x;

    def forward_mmse(self, x):
        # x - of shape [b,c,h,w]
        b,c,h,w = x.shape;

        # Acquisition
        x = x.view(b*c,h*w); # shape x = [b*c,h*w] = [b*c,N]
        x_0 = torch.zeros_like(x).to(x.device);
        x = self.Acq(x); #  shape x = [b*c, 2*M]

        # Preprocessing
        sigma_noi = self.PreP.sigma(x);
        x = self.PreP(x, self.Acq.FO); # shape x = [b*c, M]

        # Data consistency layer
        # measurements to the image domain 
        x = self.DC_layer(x, x_0, sigma_noi, self.Acq.FO); # shape x = [b*c, N]

        # Image-to-image mapping via convolutional networks 
        # Image domain denoising 
        x = x.view(b*c,1,h,w);
        return x;
    
    def reconstruct(self, x, h = None, w = None):
        # x - of shape [b,c, 2M]
        b, c, M2 = x.shape;
        
        if h is None :
            h = int(np.sqrt(self.Acq.FO.N))         
        
        if w is None :
            w = int(np.sqrt(self.Acq.FO.N))       
        
        # MMSE reconstruction    
        x = self.reconstruct_meas2im(x, h, w)
        
        # Image-to-image mapping via convolutional networks 
        # Image domain denoising 
        x = x.view(b*c,1,h,w)
        x = self.Denoi(x)   # shape stays the same
        x = x.view(b,c,h,w)
        return x;
    
    def reconstruct2(self, x):
        """
        input x is of shape [b*c, 2M]
        """
        # Measurement to image domain mapping
        x = self.reconstruct_meas2im2(x)         # shape x = [b*c,1,h,w]
        
        # Image domain denoising
        x = self.Denoi(x)                       # shape stays the same
        return x
    
    def reconstruct_meas2im2(self, x):
        # x of shape [b*c, 2M]
        bc, _ = x.shape
    
        # Preprocessing
        sigma_noi = self.PreP.sigma(x)
        x = self.PreP(x, self.Acq.FO) # shape x = [b*c, M]
        x_0 = torch.zeros((bc, self.Acq.FO.N)).to(x.device)
        #x_0 = torch.zeros_like(x).to(x.device)
        
        # measurements to image domain processing
        x = self.DC_layer(x, x_0, sigma_noi, self.Acq.FO) # shape x = [b*c, N]
        x = x.view(bc,1,self.Acq.FO.h, self.Acq.FO.w)   # shape x = [b*c,1,h,w]
        
        return x
    
    def reconstruct_meas2im(self, x, h = None, w = None):
        # x - of shape [b,c, 2M]
        b, c, M2 = x.shape;
        
        if h is None :
            h = int(np.sqrt(self.Acq.FO.N))   
        
        if w is None :
            w = int(np.sqrt(self.Acq.FO.N))
        
        x = x.view(b*c, M2)
        x_0 = torch.zeros((b*c, self.Acq.FO.N)).to(x.device)

        # Preprocessing
        sigma_noi = self.PreP.sigma(x)
        x = self.PreP(x, self.Acq.FO) # shape x = [b*c, M]

        # Data consistency layer
        # measurements to the image domain 
        x = self.DC_layer(x, x_0, sigma_noi, self.Acq.FO) # shape x = [b*c, N]

        x = x.view(b,c,h,w)
        return x
    
    def reconstruct_expe(self, x, h = None, w = None):
        # x - of shape [b,c, 2M]
        bc, _ = x.shape;
        
        if h is None:
            h = int(np.sqrt(self.Acq.FO.N))   
        
        if w is None:
            w = int(np.sqrt(self.Acq.FO.N))
        
        #x = x.view(bc, M2)
        x_0 = torch.zeros((bc, self.Acq.FO.N)).to(x.device)

        # Preprocessing experimental data
        #var_noi = self.PreP.sigma(x)
        #x, N0_est = self.PreP.forward_expe(x, self.Acq.FO) # shape x = [b*c, M]
        
        var_noi = self.PreP.sigma(x)
        x, N0_est = self.PreP.forward_expe(x, self.Acq.FO) # shape x = [b*c, M]
        #var_noi = torch.div(var_noi, N0_est**2)
        
        # Measurement to image domain 
        x = self.DC_layer(x, x_0, var_noi, self.Acq.FO) # shape x = [b*c, N]
        #x = x.view(b,c,h,w)
        
        # Image domain denoising 
        x = x.view(bc,1,h,w)
        x = self.Denoi(x)   # shape stays the same
        #x = x.view(b,c,h,w)
        
        # Denormalization
        N0_est = N0_est.view(bc,1,1,1)
        N0_est = N0_est.expand(bc,1,h,w)
        x = (x+1)*N0_est/2  
        
        return x
    
    
    def reconstruct_expe2(self, x, h = None, w = None):
        # x - of shape [b,c, 2M]
        bc, _ = x.shape;
        
        if h is None:
            h = int(np.sqrt(self.Acq.FO.N))   
        
        if w is None:
            w = int(np.sqrt(self.Acq.FO.N))
        
        #x = x.view(bc, M2)
        x_0 = torch.zeros((bc, self.Acq.FO.N)).to(x.device)

        # Preprocessing experimental data
        #var_noi = self.PreP.sigma(x)
        #x, N0_est = self.PreP.forward_expe(x, self.Acq.FO) # shape x = [b*c, M]
        
        var_noi = self.PreP.sigma_expe(x, gain=1, mudark=700, sigdark=17)
        x, N0_est = self.PreP.forward_expe(x, self.Acq.FO) # shape x = [b*c, M]
        N0_div = N0_est.view(bc,1).expand(bc,self.Acq.FO.M)
        var_noi = torch.div(var_noi, N0_div**2)
        
        # Measurement to image domain 
        x = self.DC_layer(x, x_0, var_noi, self.Acq.FO) # shape x = [b*c, N]
        #x = x.view(b,c,h,w)
        
        # Image domain denoising 
        x = x.view(bc,1,h,w)
        x = self.Denoi(x)   # shape stays the same
        #x = x.view(b,c,h,w)
        
        # Denormalization
        N0_est = N0_est.view(bc,1,1,1)
        N0_est = N0_est.expand(bc,1,h,w)
        x = (x+1)*N0_est/2  
        
        return x


# ===========================================================================================
class Pinv_Net(nn.Module):
# ===========================================================================================
    def __init__(self, Acq, PreP, DC_layer, Denoi):
        super().__init__()
        self.Acq = Acq; # must be a split operator for now
        self.PreP = PreP;
        self.DC_layer = DC_layer; # must be Pinv
        self.Denoi = Denoi;

    def forward(self, x):
        # x is of shape [b,c,h,w]
        b,c,_,_ = x.shape

        # Acquisition
        x = x.view(b,c,self.Acq.FO.N) 
        x = x.view(b*c,self.Acq.FO.N)       # shape x = [b*c,h*w] = [b*c,N]
        x = self.Acq(x)                     # shape x = [b*c, 2*M]

        # Reconstruction 
        x = self.reconstruct(x)             # shape x = [bc, 1, h,w]
        x = x.view(b,c,self.Acq.FO.h, self.Acq.FO.w)
        
        return x

    def forward_meas2im(self, x):
        # x is of shape [b,c,h,w]
        b,c,_,_ = x.shape

        # Acquisition
        x = x.view(b,c,self.Acq.FO.N) 
        x = x.view(b*c,self.Acq.FO.N)           # shape x = [b*c,h*w] = [b*c,N]
        x = self.Acq(x)                         # shape x = [b*c, 2*M]

        # Reconstruction 
        x = self.reconstruct_meas2im(x)         # shape x = [bc,1,h,w]
        x = x.view(b,c,self.Acq.FO.h, self.Acq.FO.w)
        
        return x

    def reconstruct(self, x):
        """
        input x is of shape [b*c, 2M]
        """
        # Measurement to image domain mapping
        x = self.reconstruct_meas2im(x)         # shape x = [b*c,1,h,w]
        
        # Image domain denoising
        x = self.Denoi(x)                       # shape stays the same
        
        return x

    def reconstruct_meas2im(self, x):
        # x of shape [b*c, 2M]
        bc, _ = x.shape
    
        # Preprocessing
        x = self.PreP(x, self.Acq.FO) # shape x = [b*c, M]
    
        # measurements to image domain processing
        x = self.DC_layer(x, self.Acq.FO)               # shape x = [b*c,N]
        x = x.view(bc,1,self.Acq.FO.h, self.Acq.FO.w)   # shape x = [b*c,1,h,w]
        
        return x


    def reconstruct_expe(self, x):
        """
        output image is denormalized with units of photon counts
        """
        # x of shape [b*c, 2M]
        bc, _ = x.shape
    
        # Preprocessing
        x, N0_est = self.PreP.forward_expe(x, self.Acq.FO) # shape x = [b*c, M]
        
        print(N0_est)
    
        # measurements to image domain processing
        x = self.DC_layer(x, self.Acq.FO)               # shape x = [b*c,N]
        x = x.view(bc,1,self.Acq.FO.h, self.Acq.FO.w)   # shape x = [b*c,1,h,w]

        # Image domain denoising
        x = self.Denoi(x)                               # shape x = [b*c,1,h,w]
        
        print(x.max())
        
        # Denormalization 
        x = self.PreP.denormalize_expe(x, N0_est, self.Acq.FO.h, self.Acq.FO.w)
        
        return x
    
# ===========================================================================================
class DC2_Net(Pinv_Net):
# ===========================================================================================
    def __init__(self, Acq, PreP, DC_layer, Denoi):
        super().__init__(Acq, PreP, DC_layer, Denoi)

    def reconstruct_meas2im(self, x):
        # x of shape [b*c, 2M]
        bc, _ = x.shape
    
        # Preprocessing
        var_noi = self.PreP.sigma(x)
        x = self.PreP(x, self.Acq.FO) # shape x = [b*c, M]
    
        # measurements to image domain processing
        x_0 = torch.zeros((bc, self.Acq.FO.N)).to(x.device)
        x = self.DC_layer(x, x_0, var_noi, self.Acq.FO)
        x = x.view(bc,1,self.Acq.FO.h, self.Acq.FO.w)   # shape x = [b*c,1,h,w]
        
        return x        
        
    def reconstruct_expe(self, x):
        """
        The output images are denormalized, i.e., they have units of photon counts. 
        The estimated image intensity N0 is used for both normalizing the raw 
        data and computing the variance of the normalized data.
        """
        # x of shape [b*c, 2M]
        bc, _ = x.shape
        
        # Preprocessing expe
        var_noi = self.PreP.sigma_expe(x)
        x, N0_est = self.PreP.forward_expe(x, self.Acq.FO) # x <- x/N0_est
        x = x/self.PreP.gain
        norm = self.PreP.gain*N0_est
        
        # variance of preprocessed measurements
        var_noi = torch.div(var_noi, (norm.view(-1,1).expand(bc,self.Acq.FO.M))**2)
    
        # measurements to image domain processing
        x_0 = torch.zeros((bc, self.Acq.FO.N)).to(x.device)
        x = self.DC_layer(x, x_0, var_noi, self.Acq.FO)
        x = x.view(bc,1,self.Acq.FO.h, self.Acq.FO.w)       # shape x = [b*c,1,h,w]

        # Image domain denoising
        x = self.Denoi(x)                                  # shape x = [b*c,1,h,w]
        
        # Denormalization 
        x = self.PreP.denormalize_expe(x, norm, self.Acq.FO.h, self.Acq.FO.w)
        
        return x

# ===========================================================================================
class MoDL(nn.Module):
# ===========================================================================================
    def __init__(self, Acq, PreP, DC_layer, Denoi, n_iter):
        super().__init__()
        self.Acq = Acq; 
        self.PreP = PreP;
        self.DC_layer = DC_layer; #must be a non-generalized Tikhonov
        self.Denoi = Denoi;
        self.n_iter = n_iter

    def forward(self, x):
        # x - of shape [b,c,h,w]
        b,c,h,w = x.shape;
        
        # Acquisition
        x = x.view(b*c,h*w); # shape x = [b*c,h*w] = [b*c,N]
        x_0 = torch.zeros_like(x).to(x.device);
        x = self.Acq(x); #  shape x = [b*c, 2*M]

        # Preprocessing
        m = self.PreP(x, self.Acq.FO); # shape x = [b*c, M]

        # Data consistency layer
        # measurements to the image domain
        for i in range(self.n_iter):
            x = self.DC_layer(m, x_0, self.Acq.FO); # shape x = [b*c, N]
            # Image-to-image mapping via convolutional networks 
            # Image domain denoising 
            x = x.view(b*c,1,h,w);
            x = self.Denoi(x); # shape stays the same
            x = x.view(b*c,h*w); # shape x = [b*c,h*w] = [b*c,N]
            x_0 = x;
        x = self.DC_layer(m, x_0, self.Acq.FO); # shape x = [b*c, N]
        x = x.view(b,c,h,w);
        return x;

    def forward_mmse(self, x):
        # x - of shape [b,c,h,w]
        b,c,h,w = x.shape;

        # Acquisition
        x = x.view(b*c,h*w); # shape x = [b*c,h*w] = [b*c,N]
        x_0 = torch.zeros_like(x).to(x.device);
        x = self.Acq(x); #  shape x = [b*c, 2*M]

        # Preprocessing
        x = self.PreP(x, self.Acq.FO); # shape x = [b*c, M]

        # Data consistency layer
        # measurements to the image domain 
        x = self.DC_layer(x, x_0, self.Acq.FO); # shape x = [b*c, N]

        # Image-to-image mapping via convolutional networks 
        # Image domain denoising 
        x = x.view(b*c,1,h,w);
        return x;

    def reconstruct(self, x, h = None, w = None):
        # x - of shape [b,c, 2M]
        b, c, M2 = x.shape;
        
        if h is None :
            h = int(np.sqrt(self.Acq.FO.N));           
        
        if w is None :
            w = int(np.sqrt(self.Acq.FO.N));        
        
        x = x.view(b*c, M2)
        x_0 = torch.zeros((b*c, self.Acq.FO.N)).to(x.device);

        # Preprocessing
        sigma_noi = self.PreP.sigma(x);
        m = self.PreP(x, self.Acq.FO); # shape x = [b*c, M]
        # Data consistency layer
        # measurements to the image domain
        for i in range(self.n_iter):
            x = self.DC_layer(m, x_0, self.Acq.FO); # shape x = [b*c, N]
            # Image-to-image mapping via convolutional networks 
            # Image domain denoising 
            x = x.view(b*c,1,h,w);
            x = self.Denoi(x); # shape stays the same
            x = x.view(b*c,h*w); # shape x = [b*c,h*w] = [b*c,N]
            x_0 = x;

        x = self.DC_layer(m, x_0, self.Acq.FO); # shape x = [b*c, N]
        x = x.view(b,c,h,w);
        return x;


# ===========================================================================================
class EM_net(nn.Module):
# ===========================================================================================
    def __init__(self, Acq, PreP, DC_layer, Denoi, n_iter, est_var = True):
        super().__init__()
        self.Acq = Acq; 
        self.PreP = PreP;
        self.DC_layer = DC_layer; # must be a tikhonov-list
        self.Denoi = Denoi; # Must be a denoi-list
        self.n_iter = n_iter

    def forward(self, x):
        # x - of shape [b,c,h,w]
        b,c,h,w = x.shape;
        
        # Acquisition
        x = x.view(b*c,h*w); # shape x = [b*c,h*w] = [b*c,N]
        x_0 = torch.zeros_like(x).to(x.device);
        x = self.Acq(x); #  shape x = [b*c, 2*M]

        # Preprocessing
        var_noi = self.PreP.sigma(x);
        m = self.PreP(x, self.Acq.FO); # shape x = [b*c, M]

        # Data consistency layer
        # measurements to the image domain
        for i in range(self.n_iter):
            if self.est_var :
                var_noi = self.PreP.sigma_from_image(x, self.Acq.FO);
            x = self.DC_layer(m, x_0, self.Acq.FO, iterate = i); # shape x = [b*c, N]
            # Image-to-image mapping via convolutional networks 
            # Image domain denoising 
            x = x.view(b*c,1,h,w);
            x = self.Denoi(x, iterate = i); # shape stays the same
            x = x.view(b*c,h*w); # shape x = [b*c,h*w] = [b*c,N]
            x_0 = x;
        x = x.view(b,c,h,w);
        return x;







## ==================================================================================
#class Tikhonov_cg_test(nn.Module):
## ==================================================================================
#    def __init__(self, FO, n_iter = 5, mu = 0.1, eps = 1e-6):
#        super().__init__()
#        # FO = Forward Operator - Works for ANY forward operator
#        self.n_iter = n_iter;
#        self.mu = nn.Parameter(torch.tensor([float(mu)], requires_grad=True)) #need device maybe?
#        # if user wishes to keep mu constant, then he can change requires gard to false 
#        self.eps = eps;
#        self.FO = FO
#
#    def A(self,x, FO):
#        return FO.Forward_op(FO.adjoint(x)) + self.mu*x
#
#    def CG(self, y, FO, shape, device):
#        x = torch.zeros(shape).to(device); 
#        r = y - self.A(x, FO);
#        c = r.clone()
#        kold = torch.sum(r * r)
#        a = torch.ones((1));
#        for i in range(self.n_iter): 
#            if a>self.eps : # Necessary to avoid numerical issues with a = 0 -> a = NaN
#                Ac = self.A(c, FO)
#                cAc =  torch.sum(c * Ac)
#                a =  kold / cAc
#                x += a * c
#                r -= a * Ac
#                k = torch.sum(r * r)
#                b = k / kold
#                c = r + b * c
#                kold = k
#        return x
#        
#    def forward(self, x, x_0, FO):
#        # x - input (b*c, M) - measurement vector
#        # x_0 - input (b*c, N) - previous estimate
#        # z - output (b*c, N)
#        # n_step steps of Conjugate gradient to solve \|Ax-b\|^2 + mu \|x - x_0\|^2
#        y = x-FO.Forward_op(x_0);
#        print(id(FO))
#        print(FO.Hsub.weight.data.data_ptr())
#        x = self.CG(y, FO, x.shape, x.device);
#        x = x_0 + FO.adjoint(x)
#        return x
#












#
## ==================================================================================
#class Split_Forward_operator_pylops(Split_Forward_operator):
## ==================================================================================
## Pylops compatible split forward operator 
#    def __init__(self, Hsub, device = "cpu"):           
#        # [H^+, H^-]
#        super().__init__(Hsub)
#        self.Op = LinearOperator(aslinearoperator(Hsub), device = device, dtype = torch.float32)
#

# DOES NOT WORK YET BECAUSE MatrixLinearOperator are not subscriptable
#
## ==================================================================================
#class Tikhonov_cg_pylops(nn.Module):
## ==================================================================================
#    def __init__(self, FO, n_iter = 5, mu = 0.1):
#        super().__init__()
#        # FO = Forward Operator - Needs to be pylops compatible!!
#        #-- Pseudo-inverse to determine levels of noise.
#        # Not sure about the support of learnable mu!!! (to be tested)
#        self.FO = FO;
#        self.mu = mu;
#        self.n_iter = n_iter
#
#    def A(self):
#        print(type(self.FO.Op))
#        # self.FO.Op.H NOT WORKING FOR NOW - I believe it's a bug, but here it isa
#        #
#        #File ~/.conda/envs/spyrit-env/lib/python3.8/site-packages/pylops_gpu/LinearOperator.py:336, in LinearOperator._adjoint(self)
#        #    334 def _adjoint(self):
#        #    335     """Default implementation of _adjoint; defers to rmatvec."""
#        #--> 336     shape = (self.shape[1], self.shape[0])
#        #    337     return _CustomLinearOperator(shape, matvec=self.rmatvec,
#        #    338                                  rmatvec=self.matvec,
#        #    339                                  dtype=self.dtype, explicit=self.explicit,
#        #    340                                  device=self.device, tocpu=self.tocpu,
#        #    341                                  togpu=self.togpu)
#        #
#        #TypeError: 'MatrixLinearOperator' object is not subscriptable
#        # Potentially needs to be improved
#        return self.FO.Op*self.FO.Op.T + self.mu*Diagonal(torch.ones(self.FO.M).to(self.FO.OP.device))
#        
#    def forward(self, x, x_0):
#        # x - input (b*c, M) - measurement vector
#        # x_0 - input (b*c, N) - previous estimate
#        # z - output (b*c, N)
#        
#        # Conjugate gradient to solve \|Ax-b\|^2 + mu \|x - x_0\|^2
#        #y = self.FO.Forward_op(x_0)-x;
#        #x,_ = cg(self.A(), y, niter = self.n_iter) #see pylops gpu conjugate gradient
#        #x = x_0 + self.FO.adjoint(x)
#        x = NormalEquationsInversion(Op = self.FO.Op, Regs = None, data = x, \
#                epsI = self.mu, x0 = x_0, device = self.FO.Op.device, \
#                **dict(niter = self.n_iter))
#        return x
#
# DOES NOT WORK YET BECAUSE MatrixLinearOperator are not subscriptable
#
## ==================================================================================
#class Generalised_Tikhonov_cg_pylops(nn.Module):
## ==================================================================================
#    def __init__(self, FO, Sigma_prior, n_steps):
#        super().__init__()
#        # FO = Forward Operator - pylops compatible! Does not allow to
#        # optimise the matrices Sigma_prior yet
#        self.FO = FO;
#        self.Sigma_prior = LinearOperator(aslinearoperator(Sigma_prior), self.FO.OP.device, dtype = self.FO.OP.dtype)
#
#    def A(self, var):
#        return self.FO.OP*self.Sigma_prior*self.FO.OP.H + Diagonal(var.to(self.FO.OP.device));
#        
#    def forward(self, x, x_0, var_noise):
#        # x - input (b*c, M) - measurement vector
#        # x_0 - input (b*c, N) - previous estimate
#        # z - output (b*c, N)
#        
#        # Conjugate gradient to solve \|Ax-b\|^2_Var_noise + \|x - x_0\|^2_Sigma_prior
#        y = self.FO.Forward_op(x_0)-x;
#        x,_ = cg(self.A(var_noise), y, niter = self.n_iter)
#        x = x_0 + self.Sigma_prior(self.FO.adjoint(x)) # to check that cast works well!!!
#        return x
#
