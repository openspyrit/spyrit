import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import fht
import copy
from ..misc.pattern_choice import Hadamard, matrix2conv, split
from collections import OrderedDict
import cv2
from scipy.stats import rankdata
from itertools import cycle;
# from function.learning.model_Had_DCAN import *



def Hadamard_Transform_Matrix(img_size):
    H = np.zeros((img_size**2, img_size**2))
    for i in range(img_size**2):
        base_function = np.zeros((img_size**2,1));
        base_function[i] = 1;
        base_function = np.reshape(base_function, (img_size, img_size));
        hadamard_function = fht.fht2(base_function);
        H[i, :] = np.reshape(hadamard_function, (1,img_size**2));
    return H

def Cov2Var(Cov):
    """
    Extracts Variance Matrix from Covarience Matrix
    """
    (Nx, Ny) = Cov.shape;
    diag_index = np.diag_indices(Nx);
    Var = Cov[diag_index];
    Var = np.reshape(Var, (int(np.sqrt(Nx)),int(np.sqrt(Nx))) );
    return Var

def Permutation_Matrix(had_mat):
    """
        Returns Permutation Matrix For The Hadamard Coefficients that ranks
        The Coefficients according to the Matrix defined By had_mat.
    """
    (nx, ny) = had_mat.shape;
    Reorder = rankdata(-had_mat, method = 'ordinal');
    Columns = np.array(range(nx*ny));
    P = np.zeros((nx*ny, nx*ny));
    P[Reorder-1, Columns] = 1;
    return P




########################################################################
# 0. Define a new type of layer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Creating custom layer for denoising
# ------------------------------------
# Diagonal denoising layer.   
# 
#
#def bmul(vec, mat, axis=0):
#    mat = mat.transpose(axis, -1)
#    return (mat * vec.expand_as(mat)).transpose(axis, -1)
#



class Denoise_layer(nn.Module):
    r"""Applies a transformation to the incoming data: :math:`y = A^2/(A^2+x) `

    Args:
        in_features: size of each input sample

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{in})`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, 1)`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
    Examples::

        >>> m = Denoise_layer(30)
        >>> input = torch.randn(128, 30)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(self, in_features):
        super(Denoise_layer, self).__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, 0, 2/math.sqrt(self.in_features))
        #nn.init.constant_(self.weight, 42);

    def forward(self, inputs):
        return tikho(inputs, self.weight)

    def extra_repr(self):
        return 'in_features={}'.format(self.in_features)



def tikho(inputs, weight):
    # type: (Tensor, Tensor) -> Tensor
    r"""
    Applies a transformation to the incoming data: :math:`y = A^2/(A^2+x)`.
    Shape:

        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(in\_features)`
        - Output: :math:`(N, in\_features)`
    """
    sigma = weight**2;
    den = sigma + inputs;
    ret = sigma/den;
    return ret




# =============================================================================
# A Unet for reconstruction
# =============================================================================

class Unet(nn.Module):
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

            return  block
    

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
    
    def forward(self, x ,b,c,h,w,x0, var):
        
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
    
class iteratif(nn.Module):
    def __init__(self,  M, n, Cov, H=None,denoi =None, n_iter = 5):
        super( iteratif,self).__init__()
        self.n = n;
        self.M = M;
        self.n_iter = n_iter
        if denoi == None:
            denoi = ConvNet()
        self.conv = denoi
        # Hadamard matrix
        if type(H)==type(None):
            H = Hadamard_Transform_Matrix(self.n)
        H = n*H; #fht hadamard transform needs to be normalized
        self.H=H
        #-- Hadamard patterns (undersampled basis)
        Var = Cov2Var(Cov)
        Perm = Permutation_Matrix(Var)
        self.Pmatfull = np.dot(Perm,H);
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Ptot = matrix2conv(self.Pmatfull);
        self.Ptot.weight.requires_grad = False;
        self.lambd = nn.Parameter(torch.tensor([0.05], requires_grad=True, device = device))
        
     
        self.Pinvtot = matrix2conv((1/n**2)*np.transpose(self.Pmatfull));
        self.Pinvtot.weight.requires_grad = False;
    def forward(self,x ,b ,c ,h ,w, x0, var):
        for i in range(self.n_iter):
            x=self.conv(x,b ,c ,h ,w ,x0, var)
            x=self.Ptot(x)
            x1=x[:,:self.M,:,:]
            x1 = torch.reshape(x1,(b,1,self.M,1))
            x0 = torch.reshape(x0,(b,1,self.M,1))
            
            x1 = torch.div((torch.mul(x0,self.lambd)+x1),(self.lambd+1))
            x1 = torch.reshape(x1,(b,self.M,1,1))
            x=torch.cat((x1,x[:,(self.M):,:,:]),1)
            x=torch.reshape(x,(b,c,h,w))
            x=self.Pinvtot(x)
            x=torch.reshape(x,(b,c,h,w))
        return x


    
class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()

    def forward(self,x ,b ,c ,h ,w,x0, var):
        return x
  

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
#        self.convnet = nn.Sequential(OrderedDict([
#                ('conv1', nn.Conv2d(1,64,kernel_size=9, stride=1, padding=4)),
#                ('relu1', nn.ReLU()),
#                ('BN1', nn.BatchNorm2d(64)),
#                ('conv2', nn.Conv2d(64,32,kernel_size=1, stride=1, padding=0)),
#                ('relu2', nn.ReLU()),
#                ('BN2', nn.BatchNorm2d(32)),
#                ('conv3', nn.Conv2d(32,1,kernel_size=5, stride=1, padding=2))
#                ]));
#        
        self.convnet = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(1,64,kernel_size=9, stride=1, padding=4)),
                ('relu1', nn.ReLU()),
                ('conv2', nn.Conv2d(64,32,kernel_size=1, stride=1, padding=0)),
                ('relu2', nn.ReLU()),
                ('conv3', nn.Conv2d(32,1,kernel_size=5, stride=1, padding=2))
                ]));
                
    def forward(self,x ,b ,c ,h ,w,x0, var):
        x = self.convnet(x)
        return x
    
class DConvNet(nn.Module):  
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
    def forward(self,x ,b ,c ,h ,w,x0, var):
        x = self.convnet(x)
        return x

    
class sn_dp_iteratif(nn.Module):
    def __init__(self,  M, n, Cov, H=None,denoi =None, n_iter = 5):
        super(sn_dp_iteratif,self).__init__()
        self.n = n;
        self.M = M;
        self.n_iter = n_iter
        if denoi == None:
            denoi = ConvNet()
        self.conv = denoi
        # Hadamard matrix
        if type(H)==type(None):
            H = Hadamard_Transform_Matrix(self.n)
        H = n*H; #fht hadamard transform needs to be normalized
        self.H=H
        #-- Hadamard patterns (undersampled basis)
        Var = Cov2Var(Cov)
        Perm = Permutation_Matrix(Var)
        self.Pmatfull = np.dot(Perm,H);
        self.Pinvtot = matrix2conv((1/n**2)*np.transpose(self.Pmatfull));
  
        self.Pmat = self.Pmatfull[:M,:];
        self.Pconv = matrix2conv(self.Pmat);


        self.denoise_layer = Denoise_layer(M);
        self.completion_layer = nn.Linear(M, n**2-M, False);



    def set_layers(self, denoi,Cov):
        """
            Allows initalisation of reconstructor at specified values of 
            Denoised layer, denoiser, and completion layer
            """
        self.conv = denoi;
    
        diag_index = np.diag_indices(self.n**2);
        Sigma = Cov[diag_index];
        Sigma = Sigma[:self.M]
        self.denoise_layer.weight.data = torch.from_numpy(Sigma)
    
        Sig_1 = Cov[:self.M,:self.M]
        Sig_21 = Cov[self.M:,:self.M]
        Sig_sc = np.dot(Sig_21, np.linalg.inv(Sig_1));
    
        self.completion_layer.weight.data = torch.from_numpy(Sig_sc)
    

    def forward(self,x ,b ,c ,h, w, x0, var):
        for i in range(self.n_iter):
            x = self.conv(x, b ,c ,h ,w ,x0, var)
            z = self.Pconv(x);
            z = z.view(b*c,self.M);

            y1 = torch.mul(self.denoise_layer(var.view(b*c,self.M)), x0.view(b*c,self.M)-z);
            y1 = y1.view(b*c,1,self.M)
            y2 = self.completion_layer(y1);



            y = torch.cat((y1,y2),-1) 
            y = y.view(b*c, 1, h, w);
            y = self.Pinvtot(y);
            y = y.view(b*c, 1, h, w);
            
            x = y + x
#        x = self.conv(x, b ,c ,h ,w ,x0, var)
        return x


class sn_dp_iteratif_2(nn.Module):
    def __init__(self,  M, n, Cov, H=None,denoi =None, n_iter = 5):
        super(sn_dp_iteratif_2,self).__init__()
        self.n = n;
        self.M = M;
        self.n_iter = n_iter
        if denoi == None:
            denoi = ConvNet()
        self.conv = denoi
        # Hadamard matrix
        if type(H)==type(None):
            H = Hadamard_Transform_Matrix(self.n)
        H = n*H; #fht hadamard transform needs to be normalized
        self.H=H
        #-- Hadamard patterns (undersampled basis)
        Var = Cov2Var(Cov)
        Perm = Permutation_Matrix(Var)
        self.Pmatfull = np.dot(Perm,H);
        self.Pinvtot = matrix2conv((1/n**2)*np.transpose(self.Pmatfull));
        self.Pinvtot.weight.requires_grad = False;
        
  
        self.Pmat = self.Pmatfull[:M,:];
        self.Pconv = matrix2conv(self.Pmat);
        self.Pconv.weight.requires_grad = False


        self.denoise_layer = Denoise_layer(M);
        self.completion_layer = nn.Linear(M, n**2-M, False);



    def set_layers(self, denoi,Cov):
        """
            Allows initalisation of reconstructor at specified values of 
            Denoised layer, denoiser, and completion layer
            """
        self.conv = denoi;
    
        diag_index = np.diag_indices(self.n**2);
        Sigma = Cov[diag_index];
        Sigma = Sigma[:self.M]
        self.denoise_layer.weight.data = torch.from_numpy(Sigma)
    
        Sig_1 = Cov[:self.M,:self.M]
        Sig_21 = Cov[self.M:,:self.M]
        Sig_sc = np.dot(Sig_21, np.linalg.inv(Sig_1));
    
        self.completion_layer.weight.data = torch.from_numpy(Sig_sc)
    

    def forward(self,x ,b ,c ,h, w, x0, var):
        for i in range(self.n_iter):
            x = self.conv(x, b ,c ,h ,w ,x0, var)
            z = self.Pconv(x);
            z = z.view(b*c,self.M);

            y1 = torch.mul(self.denoise_layer(var.view(b*c,self.M)), x0.view(b*c,self.M)-z);
            y1 = y1.view(b*c,1,self.M)
            y2 = self.completion_layer(y1);



            y = torch.cat((y1,y2),-1) 
            y = y.view(b*c, 1, h, w);
            y = self.Pinvtot(y);
            y = y.view(b*c, 1, h, w);
            
            x = y + x
        x = self.conv(x, b ,c ,h ,w ,x0, var)
        return x


def Stat_net(dataloader, root, model, device, transform, save = False):
    """ 
        Computes Mean Hadamard Image over the whole dataset + 
        Covariance Matrix Amongst the coefficients
    """

    (inputs, classes) = next(iter(dataloader["train"]))
    inputs = inputs.cpu().detach().numpy();
    (batch_size, channels, nx, ny) = inputs.shape;
    tot_num = len(dataloader)*batch_size;


    transform = transform.to(device)
    c = 0;
    Cov_had = np.zeros((nx*ny, nx*ny));
    with torch.no_grad():
        for inputs,labels in dataloader["train"]:
            c +=1;
            print("batch {}/{}".format(c, len(dataloader["train"])))
            inputs = inputs.to(device);
            (batch_size, channels, nx, ny) = inputs.shape;
            output = model(inputs);
            images = transform(inputs-output)
            images = images.view(batch_size, channels, nx, ny);
            images = images.cpu().detach().numpy();
            for i in range(images.shape[0]):
                img = images[i,0,:,:];
                Norm_Variable = np.reshape(img, (nx*ny,1));
                Cov_had += Norm_Variable*np.transpose(Norm_Variable);
        Cov_had = Cov_had/(tot_num-1);
    if save :
        np.save(root+'Cov_{}x{}'.format(nx,ny)+'.npy', Cov_had)
    return Cov_had 

        
class em_dl_iteratif_2(nn.Module):
    def __init__(self,  M, n, Cov, H=None,denoi = None, n_iter = 5):
        super(em_dl_iteratif_2,self).__init__()
        self.n = n;
        self.M = M;
        self.n_iter = n_iter
        if denoi == None:
            denoi = ConvNet();
        # Hadamard matrix
        if type(H)==type(None):
            H = Hadamard_Transform_Matrix(self.n)
        H = n*H; #fht hadamard transform needs to be normalized
        self.H=H
        #-- Hadamard patterns (undersampled basis)
        Var = Cov2Var(Cov)
        Perm = Permutation_Matrix(Var)
        self.Pmatfull = np.dot(Perm,H);
        self.Pinvtot = matrix2conv((1/n**2)*np.transpose(self.Pmatfull));
  
        self.Pmat = self.Pmatfull[:M,:];
        self.Pconv = matrix2conv(self.Pmat);


        self.conv = denoi;
        denoise_list = [];
        completion_list = [];
        for i in range(n_iter):
            denoise_list.append(Denoise_layer(M));
            completion_list.append(nn.Linear(M, n**2-M, False))
        conv_list.append(denoi());

        self.denoise_list = nn.ModuleList(denoise_list);
        self.completion_list = nn.ModuleList(completion_list);
        
    def forward(self,x ,b ,c ,h, w, x0, var):
        for i in range(self.n_iter):
            x = self.conv(x, b ,c ,h ,w ,x0, var)
            z = self.Pconv(x);
            z = z.view(b*c,self.M);
#
#            print(var.dtype)
#            print(var.device)
#            print(self.denoise_list[i].weight.dtype)
#            print(self.denoise_list[i].weight.device)
#
            y1 = torch.mul(self.denoise_list[i](var.view(b*c,self.M)), x0.view(b*c,self.M)-z);
            y1 = y1.view(b*c,1,self.M)
            y2 = self.completion_list[i](y1);

            y = torch.cat((y1,y2),-1) 
            y = y.view(b*c, 1, h, w);
            y = self.Pinvtot(y);
            y = y.view(b*c, 1, h, w);
            
            x = y + x
        x = self.conv(x, b ,c ,h ,w ,x0, var)
        return x

    def set_denoise_layer(self, denoise):
        self.conv = copy.deepcopy(denoise);

    def set_layers(self, em_net, Cov):
        """
            Allows initalisation of reconstructor at specified values of 
            Denoised layer, denoiser, and completion layer
            """
        self.conv = copy.deepcopy(em_net.conv);
        nb_iter = min(self.n_iter, em_net.n_iter)
        for i in range(nb_iter):
            print(i)
            self.denoise_list[i] = copy.deepcopy(em_net.denoise_list[i]);
            self.completion_list[i] = copy.deepcopy(em_net.completion_list[i]);
        diag_index = np.diag_indices(self.n**2);
        Sigma = Cov[diag_index];
        Sigma = Sigma[:self.M]
        print(len(self.denoise_list))
        print(nb_iter)
        self.denoise_list[nb_iter].weight.data = torch.from_numpy(Sigma)
    
        Sig_1 = Cov[:self.M,:self.M]
        Sig_21 = Cov[self.M:,:self.M]
        Sig_sc = np.dot(Sig_21, np.linalg.inv(Sig_1));
        self.completion_list[nb_iter].weight.data = torch.from_numpy(Sig_sc)

    def set_init_layers(self, denoi, Cov):
        """
            Allows initalisation of reconstructor at specified values of 
            Denoised layer, denoiser, and completion layer
            """
        self.conv = copy.deepcopy(denoi);
    
        diag_index = np.diag_indices(self.n**2);
        Sigma = Cov[diag_index];
        Sigma = Sigma[:self.M]
        self.denoise_list[0].weight.data = torch.from_numpy(Sigma)
    
        Sig_1 = Cov[:self.M,:self.M]
        Sig_21 = Cov[self.M:,:self.M]
        Sig_sc = np.dot(Sig_21, np.linalg.inv(Sig_1));
    
        self.completion_list[0].weight.data = torch.from_numpy(Sig_sc)
    


    def fix_layers(self, n_iter):
        """
            Allows to make the parameters before a certain iteration as non-learnable
        """
#        self.conv.weight.requires_grad = False;
#        self.conv.bias.requires_grad = False;
        for i in range(n_iter):
            self.denoise_list[i].weight.requires_grad = False;
            self.completion_list[i].weight.requires_grad = False;

    def all_layers_trainable(self):
        """
            Sets all parameters in the neural network as trainable parameters """
        for i in range(self.n_iter):
            self.denoise_list[i].weight.requires_grad = True;
            self.completion_list[i].weight.requires_grad = True;

class em_dl_iteratif_3(nn.Module):
    def __init__(self,  M, n, Cov, H=None,denoi =None, n_iter = 5):
        super(em_dl_iteratif_3,self).__init__()
        self.n = n;
        self.M = M;
        self.n_iter = n_iter
        if denoi == None:
            denoi = ConvNet
        # Hadamard matrix
        if type(H)==type(None):
            H = Hadamard_Transform_Matrix(self.n)
        H = n*H; #fht hadamard transform needs to be normalized
        self.H=H
        #-- Hadamard patterns (undersampled basis)
        Var = Cov2Var(Cov)
        Perm = Permutation_Matrix(Var)
        self.Pmatfull = np.dot(Perm,H);
        self.Pinvtot = matrix2conv((1/n**2)*np.transpose(self.Pmatfull));
  
        self.Pmat = self.Pmatfull[:M,:];
        self.Pconv = matrix2conv(self.Pmat);



        denoise_list = [];
        completion_list = [];
        conv_list = [];
        for i in range(n_iter):
            denoise_list.append(Denoise_layer(M));
            completion_list.append(nn.Linear(M, n**2-M, False))
            conv_list.append(denoi());
        conv_list.append(denoi());

        self.denoise_list = nn.ModuleList(denoise_list);
        self.completion_list = nn.ModuleList(completion_list);
        self.conv = nn.ModuleList(conv_list);

    def forward(self,x ,b ,c ,h, w, x0, var):
        for i in range(self.n_iter):
            x = self.conv[i](x, b ,c ,h ,w ,x0, var)
            z = self.Pconv(x);
            z = z.view(b*c,self.M);
#
#            print(var.dtype)
#            print(var.device)
#            print(self.denoise_list[i].weight.dtype)
#            print(self.denoise_list[i].weight.device)
#
            y1 = torch.mul(self.denoise_list[i](var.view(b*c,self.M)), x0.view(b*c,self.M)-z);
            y1 = y1.view(b*c,1,self.M)
            y2 = self.completion_list[i](y1);

            y = torch.cat((y1,y2),-1) 
            y = y.view(b*c, 1, h, w);
            y = self.Pinvtot(y);
            y = y.view(b*c, 1, h, w);
            
            x = y + x
        x = self.conv[-1](x, b ,c ,h ,w ,x0, var)
        return x

    def set_denoise_layer(self, denoise, n_iter):
        self.conv[n_iter] = copy.deepcopy(denoise);

    def set_layers(self, em_net, Cov):
        """
            Allows initalisation of reconstructor at specified values of 
            Denoised layer, denoiser, and completion layer
            """

        nb_iter = min(self.n_iter, em_net.n_iter)
        for i in range(nb_iter):
            print(i)
            self.conv[i] = copy.deepcopy(em_net.conv[i]);
            self.denoise_list[i] = copy.deepcopy(em_net.denoise_list[i]);
            self.completion_list[i] = copy.deepcopy(em_net.completion_list[i]);
   

        diag_index = np.diag_indices(self.n**2);
        Sigma = Cov[diag_index];
        Sigma = Sigma[:self.M]
        print(len(self.denoise_list))
        print(nb_iter)
        self.denoise_list[nb_iter].weight.data = torch.from_numpy(Sigma)
    
        Sig_1 = Cov[:self.M,:self.M]
        Sig_21 = Cov[self.M:,:self.M]
        Sig_sc = np.dot(Sig_21, np.linalg.inv(Sig_1));
        self.completion_list[nb_iter].weight.data = torch.from_numpy(Sig_sc)

    def set_init_layers(self, denoi, Cov):
        """
            Allows initalisation of reconstructor at specified values of 
            Denoised layer, denoiser, and completion layer
            """
        self.conv[0] = copy.deepcopy(denoi);
    
        diag_index = np.diag_indices(self.n**2);
        Sigma = Cov[diag_index];
        Sigma = Sigma[:self.M]
        self.denoise_list[0].weight.data = torch.from_numpy(Sigma)
    
        Sig_1 = Cov[:self.M,:self.M]
        Sig_21 = Cov[self.M:,:self.M]
        Sig_sc = np.dot(Sig_21, np.linalg.inv(Sig_1));
    
        self.completion_list[0].weight.data = torch.from_numpy(Sig_sc)
    


    def fix_layers(self, n_iter):
        """
            Allows to make the parameters before a certain iteration as non-learnable
        """
        for i in range(n_iter):
            for param in self.conv[i].parameters():
                param.requires_grad = False
            self.denoise_list[i].weight.requires_grad = False;
            self.completion_list[i].weight.requires_grad = False;

    def all_layers_trainable(self):
        """
            Sets all parameters in the neural network as trainable parameters """
        for i in range(self.n_iter):
            for param in self.conv[i].parameters():
                param.requires_grad = True;
            self.denoise_list[i].weight.requires_grad = True;
            self.completion_list[i].weight.requires_grad = True;
        for param in self.conv[-1].parameters():
            param.requires_grad = True;


 

class NeumannNet(nn.Module):
    def __init__(self,  M, n, Cov, H=None,denoi =None,iterations =5,  eta_initial_val=0.0001):
        super(NeumannNet,self).__init__()

        self.n = n;
        self.M = M;
        self.iterations = iterations;
        if denoi is None:
            denoi = ConvNet()
        # Hadamard matrix
        if type(H)==type(None):
            H = Hadamard_Transform_Matrix(self.n)
        H = n*H; #fht hadamard transform needs to be normalized
        self.H=H;
        #-- Hadamard patterns (undersampled basis)
        Var = Cov2Var(Cov)
        Perm = Permutation_Matrix(Var)
        self.Pmatfull = np.dot(Perm,H);
        self.Pinvtot = matrix2conv((1/n**2)*np.transpose(self.Pmatfull));
        self.Pinvtot.weight.requires_grad = False;
        
  
        self.Pmat = self.Pmatfull[:M,:];
        Gramian_mat = np.dot(np.transpose(self.Pmat), self.Pmat)
        
        self.Gramian = nn.Linear(n**2,n**2, False)
        self.Gramian.weight.data=torch.from_numpy(Gramian_mat);
        self.Gramian.weight.data=self.Gramian.weight.data.float();
        self.Gramian.weight.requires_grald=False;
        
        self.nonlinear_op = denoi;

        self.register_parameter(name='eta', param=torch.nn.Parameter(torch.tensor(eta_initial_val), requires_grad=True))    

    def _linear_op(self, x):
        return self.linear_op.forward(x)
#    def _linear_adjoint(self, x):
#        return self.linear_op.adjoint(x)
#    def initial_point(self, y):
#        return self._linear_adjoint(y)
        
    def gramian(self, y ,b ,c ,h, w):
        y = y.view(b*c, 1, h*w);
        y = self.Gramian(y);
        y = y.view(b*c, 1, h,w);
        return y;
        
    def single_block(self, x, b ,c ,h, w, x0, var):
        return x - self.eta * self.gramian(x ,b ,c ,h, w) - self.nonlinear_op(x, b ,c ,h, w, x0, var);
    
    def forward(self, x ,b ,c ,h, w, x0, var):
    # Needs to use Map to image as Pinv!!!!! Also reduces the need to use initial_point
#        initial_point = self.eta *self.n**2* x;
        initial_point = self.eta *self.n**2*x;
        running_term = initial_point;
        accumulator = initial_point;

        for bb in range(self.iterations):
            running_term = self.single_block(running_term, b ,c ,h, w, x0, var)
            accumulator = accumulator + running_term

        return accumulator
    


