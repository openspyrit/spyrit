# -----------------------------------------------------------------------------
#   This software is distributed under the terms
#   of the GNU Lesser General  Public Licence (LGPL)
#   See LICENSE.md for further details
# -----------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import init
import os



##############################
#      Convolutional 
#   Gated Recurrent Units
##############################


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, dtype):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        
        
         Copyright (c) 2019 Jin Li
        
        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:
        
        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.
        
        """
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype

        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=input_dim+hidden_dim,
                              out_channels=self.hidden_dim, # for candidate neural memory
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype))

    def forward(self, input_tensor, h_cur):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)
        #cnm = F.relu(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next


class pred_frame(nn.Module):
    def __init__(self) :
        super(pred_frame, self).__init__()

    def forward(self, x):
        return x;

    def evaluate(self, x):
        return x;


class pred_net_0(nn.Module):
    def __init__(self, img_size, hidden_dim=64, kernel_sizes = (5,5) , dtype = torch.cuda.FloatTensor) :
        super(pred_net_0, self).__init__()
        self.n = img_size;
        self.hidden_dim = hidden_dim;
        self.convGRU = ConvGRUCell((img_size, img_size), 1,hidden_dim ,kernel_sizes, True, dtype);
        self.conv_end = nn.Conv2d(hidden_dim,1,kernel_size=5, stride=1, padding=2);

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape;
        hidden_state = self.convGRU.init_hidden(batch_size);
        final_frame = x[:,-1,:,:,:];
        for t in range(seq_length):
            input_frame = x[:,t,:,:,:]
            hidden_state = self.convGRU(input_tensor = input_frame, h_cur = hidden_state);
        y = final_frame+self.conv_end(hidden_state);
        return y;

    def evaluate(self, x):
        batch_size, seq_length, c, h, w = x.shape;
        hidden_state = self.convGRU.init_hidden(batch_size);
        output = [];
        for t in range(seq_length):
            input_frame = x[:,t,:,:,:];
            hidden_state = self.convGRU(input_tensor = input_frame, h_cur = hidden_state);
            y = input_frame+self.conv_end(hidden_state);
            output.append(y);
        y = torch.stack(output, dim=1);
        return y;

class pred_net_2(nn.Module):
    def __init__(self, img_size, hidden_dim=4, kernel_sizes = (3,3) , dtype = torch.cuda.FloatTensor) :
        super(pred_net_2, self).__init__()
        self.n = img_size;
        self.c = hidden_dim;

        self.Mp_1 = nn.MaxPool2d(kernel_size = 2);
        self.Mp_2 = nn.MaxPool2d(kernel_size = 2);
        self.Mp_3 = nn.MaxUnpool2d(kernel_size = 2);
        self.Mp_5 = nn.MaxUnpool2d(kernel_size = 2);
        self.Bn_1 = nn.BatchNorm2d(hidden_dim);
        self.Bn_2 = nn.BatchNorm2d(2*hidden_dim);
        self.Bn_4 = nn.BatchNorm2d(4*hidden_dim);
        self.Bn_5 = nn.BatchNorm2d(2*hidden_dim);
        self.Bn_6 = nn.BatchNorm2d(hidden_dim);
        self.convGRU_1 = ConvGRUCell((img_size, img_size), 1,hidden_dim ,kernel_sizes, True, dtype);
        self.convGRU_2 = ConvGRUCell((img_size//2, img_size//2), hidden_dim ,2*hidden_dim ,kernel_sizes, True, dtype);
        self.convGRU_3 = ConvGRUCell((img_size//4, img_size//4), 2*hidden_dim ,4*hidden_dim ,kernel_sizes, True, dtype);
        self.convGRU_4 = ConvGRUCell((img_size//2, img_size//2), 4*hidden_dim ,4*hidden_dim ,kernel_sizes, True, dtype);
        self.convGRU_5 = ConvGRUCell((img_size//2, img_size//2), 4*hidden_dim ,2*hidden_dim ,kernel_sizes, True, dtype);
        self.convGRU_6 = ConvGRUCell((img_size, img_size), 2*hidden_dim ,hidden_dim ,kernel_sizes, True, dtype);
        self.conv_end = nn.Conv2d(hidden_dim,1,kernel_size=5, stride=1, padding=2);

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape;
        hidden_state_1 = self.convGRU_1.init_hidden(batch_size);
        hidden_state_2 = self.convGRU_2.init_hidden(batch_size);
        hidden_state_3 = self.convGRU_3.init_hidden(batch_size);
        hidden_state_4 = self.convGRU_4.init_hidden(batch_size);
        hidden_state_5 = self.convGRU_5.init_hidden(batch_size);
        hidden_state_6 = self.convGRU_6.init_hidden(batch_size);
        final_frame = x[:,-1,:,:,:];
        for t in range(seq_length):
            inputs = x[:,t,:,:,:];
            hidden_state_1 = self.convGRU_1(input_tensor = inputs, h_cur = hidden_state_1);
            y = self.Bn_1(hidden_state_1);
            y = self.Mp_1(y);
            hidden_state_2 = self.convGRU_2(input_tensor = y, h_cur = hidden_state_2);
            c_2 = self.Bn_2(hidden_state_2);
            y = self.Mp_2(c_2);
            hidden_state_3 = self.convGRU_3(input_tensor = y, h_cur = hidden_state_3);
            
            y = F.interpolate(hidden_state_3, size = (h//2,w//2), mode = 'nearest');
            hidden_state_4 = self.convGRU_4(input_tensor = y, h_cur = hidden_state_4);
            y = self.Bn_4(hidden_state_4);
            hidden_state_5 = self.convGRU_5(input_tensor = y, h_cur = hidden_state_5);
            y = self.Bn_5(hidden_state_5);
            y = c_2+y;
            #y = self.Mp_5(y, ind_1);
            y = F.interpolate(y, size = (h,w), mode = 'nearest');
            hidden_state_6 = self.convGRU_6(input_tensor = y, h_cur = hidden_state_6);
            y = self.Bn_6(hidden_state_6);
            
        y = final_frame+self.conv_end(y);
        return y;

    def evaluate(self, x):
        batch_size, seq_length, c, h, w = x.shape;
        hidden_state_1 = self.convGRU_1.init_hidden(batch_size);
        hidden_state_2 = self.convGRU_2.init_hidden(batch_size);
        hidden_state_3 = self.convGRU_3.init_hidden(batch_size);
        hidden_state_4 = self.convGRU_4.init_hidden(batch_size);
        hidden_state_5 = self.convGRU_5.init_hidden(batch_size);
        hidden_state_6 = self.convGRU_6.init_hidden(batch_size);
        output = [];
        for t in range(seq_length):
            inputs = x[:,t,:,:,:];
            hidden_state_1 = self.convGRU_1(input_tensor = inputs, h_cur = hidden_state_1);
            y = self.Bn_1(hidden_state_1);
            y = self.Mp_1(y);
            hidden_state_2 = self.convGRU_2(input_tensor = y, h_cur = hidden_state_2);
            c_2 = self.Bn_2(hidden_state_2);
            y = self.Mp_2(c_2);
            hidden_state_3 = self.convGRU_3(input_tensor = y, h_cur = hidden_state_3);
            
            y = F.interpolate(hidden_state_3, size = (h//2,w//2), mode = 'nearest');
            hidden_state_4 = self.convGRU_4(input_tensor = y, h_cur = hidden_state_4);
            y = self.Bn_4(hidden_state_4);
            hidden_state_5 = self.convGRU_5(input_tensor = y, h_cur = hidden_state_5);
            y = self.Bn_5(hidden_state_5);
            y = c_2+y;
            #y = self.Mp_5(y, ind_1);
            y = F.interpolate(y, size = (h,w), mode = 'nearest');
            hidden_state_6 = self.convGRU_6(input_tensor = y, h_cur = hidden_state_6);
            y = self.Bn_6(hidden_state_6);
            y = inputs+self.conv_end(y);
            output.append(y);
        y = torch.stack(output, dim=1);
        return y;




class Weight_Decay_Loss(nn.Module):
    
    def __init__(self, loss):
        super(Weight_Decay_Loss,self).__init__()
        self.loss = loss;

    def forward(self,x,y, net):
        mse=self.loss(x,y);
        return mse


