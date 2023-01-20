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
#from ..misc.disp import *
from .model_Had_DCAN import *
from ..misc.pattern_choice import Hadamard, matrix2conv


##############################
#           LSTM
##############################


class LSTM(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim, bidirectional):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, x):
        x, self.hidden_state = self.lstm(x, self.hidden_state)
        return x


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


class ConvGRU(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 dtype, batch_first=False, bias=True, return_all_layers=False):
        """
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int e.g. 256
            Number of channels of input tensor.
        :param hidden_dim: int e.g. 1024
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param num_layers: int
            Number of ConvGRU layers
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        :param alexnet_path: str
            pretrained alexnet parameters
        :param batch_first: bool
            if the first position of array is batch or not
        :param bias: bool
            Whether or not to add the bias.
        :param return_all_layers: bool
            if return hidden and cell states for all layers
        
        
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
        super(ConvGRU, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(ConvGRUCell(input_size=(self.height, self.width),
                                         input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias,
                                         dtype=self.dtype))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        :param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not
            extracted features from alexnet
        :param hidden_state:
        :return: layer_output_list, last_state_list
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # input current hidden and cell state then compute the next hidden and cell state through ConvLSTMCell forward function
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], # (b,t,c,h,w)
                                              h_cur=h)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param      
        
##############################
#         Convolutional 
#        Reconstructor
#        With Hadamard 
#        Coefficients
##############################


class RNN_Had(nn.Module):
    def __init__(self, Cov, Mean, M=333,channels = 3,hidden_dim=[64,32], kernel_sizes = [(9,9),(1,1)], n_layers = 2, dtype = torch.cuda.FloatTensor) :
        super(RNN_Had, self).__init__()
        n, ny = Mean.shape;
        self.n = n;
        self.M = M;
        self.hidden_dim = hidden_dim;
        W, b, mu1, P, H = Hadamard_stat_completion_matrices(Cov, Mean,M);
        W = (1/n)*W;
        b = (1/n)*b;
        b = b - np.dot(W,mu1);
        Patt = np.dot(P,H);
        Patt = Patt[:M,:];
        Patt = matrix2conv(Patt);
        self.Patt = Patt;
        P, T = split(Patt, 1);

        self.P = P;
        self.T = T;
        self.P.bias.requires_grad = False;
        self.P.weight.requires_grad = False;
        self.Patt.bias.requires_grad = False;
        self.Patt.weight.requires_grad = False;
        self.T.weight.requires_grad=False;
        self.T.weight.requires_grad=False;

        self.fc1 = nn.Linear(M,n**2)
        self.fc1.bias.data=torch.from_numpy(b[:,0]);
        self.fc1.bias.data=self.fc1.bias.data.float();
        self.fc1.weight.data=torch.from_numpy(W);
        self.fc1.weight.data=self.fc1.weight.data.float();
        self.fc1.bias.requires_grad = False;
        self.fc1.weight.requires_grad=False;


        self.convGRU = ConvGRU((n,n), channels,hidden_dim,kernel_sizes, n_layers, dtype, batch_first = True);
        self.conv_end = nn.Conv2d(hidden_dim[-1],1,kernel_size=5, stride=1, padding=2);


    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape;
        x =(x+1)/2;
        
        #Acquisition
        x = x.view(batch_size*seq_length *c, 1, h, w);
        x = self.P(x);
        x = F.relu(x);
        x = x.view(batch_size*seq_length*c,1, 2*self.M);

        #Pre-processing
        x = self.T(x);
        x = 2*x-torch.reshape(self.Patt(torch.ones(batch_size*seq_length*c,1, h,w).to(x.device)),(batch_size*seq_length*c,1,self.M));
        
        #Projection to the image domain
        x = self.fc1(x);

        #Post-Processing
        x = x.view([batch_size, seq_length, c, h, w]);
        x = self.convGRU(x)
        x = x[0][-1];
        x = x.view(batch_size*seq_length*c,self.hidden_dim[-1] , h, w) ;
        x = self.conv_end(x);
        x = x.view([batch_size, seq_length, c, h, w]);
        return x



class RNN_Had_noi(nn.Module):
    def __init__(self, Cov, Mean, M=333,channels = 3,hidden_dim=[64,32], kernel_sizes = [(9,9),(1,1)], No = 1e5, n_layers = 2, dtype = torch.cuda.FloatTensor) :
        super(RNN_Had_noi, self).__init__()
        n, ny = Mean.shape;
        self.n = n;
        self.M = M;
        self.No = No;
        self.hidden_dim = hidden_dim;
        W, b, mu1, P, H = Hadamard_stat_completion_matrices(Cov, Mean,M);
        W = (1/n)*W;
        b = (1/n)*b;
        b = b - np.dot(W,mu1);
        Patt = np.dot(P,H);
        Patt = Patt[:M,:];
        Patt = matrix2conv(Patt);
        self.Patt = Patt;
        P, T = split(Patt, 1);

        self.P = P;
        self.T = T;
        self.P.bias.requires_grad = False;
        self.P.weight.requires_grad = False;
        self.Patt.bias.requires_grad = False;
        self.Patt.weight.requires_grad = False;
        self.T.weight.requires_grad=False;
        self.T.weight.requires_grad=False;

        self.fc1 = nn.Linear(M,n**2)
        self.fc1.bias.data=torch.from_numpy(b[:,0]);
        self.fc1.bias.data=self.fc1.bias.data.float();
        self.fc1.weight.data=torch.from_numpy(W);
        self.fc1.weight.data=self.fc1.weight.data.float();
        self.fc1.bias.requires_grad = False;
        self.fc1.weight.requires_grad=False;


        self.convGRU = ConvGRU((n,n), channels,hidden_dim,kernel_sizes, n_layers, dtype, batch_first = True);
        self.conv_end = nn.Conv2d(hidden_dim[-1],1,kernel_size=5, stride=1, padding=2);


    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape;
        x =self.No*(x+1)/2;
        
        #Acquisition
        x = x.view(batch_size*seq_length *c, 1, h, w);
        x = self.P(x);
        x = F.relu(x);
        x = x.view(batch_size*seq_length*c,1, 2*self.M);
        x = x + torch.sqrt(x)*torch.randn_like(x);
        
        #Pre-processing
        x = self.T(x);
        x = (1/self.No)*(2*x-torch.reshape(self.Patt(torch.ones(batch_size*seq_length*c,1, h,w).to(x.device)),(batch_size*seq_length*c,1,self.M)));
        
        #Projection to the image domain
        x = self.fc1(x);

        #Post-Processing
        x = x.view([batch_size, seq_length, c, h, w]);
        x = self.convGRU(x)
        x = x[0][-1];
        x = x.view(batch_size*seq_length*c,self.hidden_dim[-1] , h, w) ;
        x = self.conv_end(x);
        x = x.view([batch_size, seq_length, c, h, w]);
        return x;




