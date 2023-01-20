# -----------------------------------------------------------------------------
#   This software is distributed under the terms
#   of the GNU Lesser General  Public Licence (LGPL)
#   See LICENSE.md for further details
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

########################################################################
# 1. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Take the DCAN network from the corresponding file
# Building up the DCAN with learned patterns as an 
# autoencoder


class Net_DCAN(nn.Module):

    def __init__(self, n=128, M=666):
        super(Net_DCAN, self).__init__()
        self.n=n
        self.M=M
        self.Patt = nn.Conv2d(1,M,kernel_size=n, stride=1, padding=0)
        self.fc1 = nn.Linear(M,n**2)

        self.recon = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,64,kernel_size=9, stride=1, padding=4)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(64,32,kernel_size=1, stride=1, padding=0)),
          ('relu2', nn.ReLU()),
          ('conv3', nn.Conv2d(32,1,kernel_size=5, stride=1, padding=2))
        ]));

       
        # Bias of the initial layer are fixed to 0 and frozen for the training
        self.Patt.bias.data=torch.zeros(M);
        self.Patt.bias.requires_grad = False;

    def forward(self, x):
        batch_size,c,h,w = x.shape;
        
        #Acquisition
        x = x.view(batch_size *c, 1, h, w);
        x = self.Patt(x);
        x = x.view(batch_size*c,1, self.M);
        
        #Projection to the image domain
        x = self.fc1(x);
        x = F.relu(x);
        x = x.view(batch_size*c, 1, h, w);

        #Post-Processing
        x = self.recon(x);
        x = x.view(batch_size,c, h, w);
        return x;


class Net_DCAN_noise(nn.Module):

    def __init__(self, n=128, M=666, No = 1e5):
        super(Net_DCAN, self).__init__()
        self.n=n;
        self.M=M;
        self.No = No;
        self.Patt = nn.Conv2d(1,M,kernel_size=n, stride=1, padding=0);
        self.fc1 = nn.Linear(M,n**2);

        self.recon = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,64,kernel_size=9, stride=1, padding=4)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(64,32,kernel_size=1, stride=1, padding=0)),
          ('relu2', nn.ReLU()),
          ('conv3', nn.Conv2d(32,1,kernel_size=5, stride=1, padding=2))
        ]));

       
        # Bias of the initial layer are fixed to 0 and frozen for the training
        self.Patt.bias.data=torch.zeros(M);
        self.Patt.bias.requires_grad = False;

    def forward(self, x):
        batch_size,c,h,w = x.shape;
        x =self.No*(x+1)/2;
        
        #Acquisition
        x = x.view(batch_size *c, 1, h, w);
        x = self.Patt(x);
        x = F.relu(x);
        x = x.view(batch_size*c,1, 2*self.M);
        x = x + torch.sqrt(x)*torch.randn_like(x);
        
        #Projection to the image domain
        x = self.fc1(x);
        x = F.relu(x);
        x = x.view(batch_size*c, 1, h, w);

        #Post-Processing
        x = self.recon(x);
        x = x.view(batch_size,c, h, w);
        return x;


class Net_DCAN_vid(nn.Module):

    def __init__(self, n=128, M=333):
        super(Net_DCAN_vid, self).__init__()
        self.n=n
        self.M=M
        self.Patt = nn.Conv2d(1,M,kernel_size=n, stride=1, padding=0)
        self.fc1 = nn.Linear(M,n**2)

        self.recon = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,64,kernel_size=9, stride=1, padding=4)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(64,32,kernel_size=1, stride=1, padding=0)),
          ('relu2', nn.ReLU()),
          ('conv3', nn.Conv2d(32,1,kernel_size=5, stride=1, padding=2))
        ]));

       
        # Bias of the initial layer are fixed to 0 and frozen for the training
        self.Patt.bias.data=torch.zeros(M);
        self.Patt.bias.requires_grad = False;

    def forward(self, x):
        """
            x is a 5 dimentional entry this time around :
            [batch_size, seq_lenght,channels, h, w ]
        """
        batch_size, seq_length, c, h, w = x.shape
        x =(x+1)/2;
        x = x.view(batch_size * seq_length*c, 1, h, w) 
        x = self.Patt(x);
        x = x.view(batch_size*seq_length*c,1, self.M);
        
        #Projection to the image domain
        x = self.fc1(x);
        x = F.relu(x);
        x = x.view(batch_size*seq_length*c, 1, h, w);

        #Post-Processing
        x = self.recon(x);

        x = x.view([batch_size, seq_length, c, h, w]);
        return x





########################################################################
# 2. Define a custom Loss function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Creating custom loss function
# ---------------------------
# Added penalization so that the weights of the encoding part are 
# Binary weights.    
#

class Binarize_Loss(nn.Module):
    
    def __init__(self,reg, loss):
        super(Binarize_Loss,self).__init__()
        self.regularizer = reg;
        self.loss = loss;

    def forward(self,x,y, net):
        
        mse=self.loss(x,y);

        # params=list(net.conv1.parameters());
        params = list(net.parameters());
        patterns = params[0];
        penalization=torch.mul((patterns+1)**2,(patterns)**2);
        totloss=self.regularizer*torch.sum(penalization)+mse;
        return totloss


