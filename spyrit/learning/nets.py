# -----------------------------------------------------------------------------
#   This software is distributed under the terms
#   of the GNU Lesser General  Public Licence (LGPL)
#   See LICENSE.md for further details
# -----------------------------------------------------------------------------

from __future__ import print_function, division
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import datetime
import copy
import pickle
######################################################################
# 1. Visualize a few images from the training set
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.


def imshow(img, title=""):
    """ 
    
    """
    plt.ion()   # interactive mode
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    if title is not None:
        plt.title(title)
    plt.pause(0.0001)


########################################################################
# 2. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# We  loop over our data iterator, feed the inputs to the
# network and optimize.

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_trainable_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_param(model):
    return sum(p.numel() for p in model.parameters())


def train_model(model, criterion, optimizer, scheduler, dataloaders, device, root, num_epochs=25,disp=False, do_checkpoint=0):
    """ Trains the pytorch model 
        """
    since = time.time()
    best_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())

    dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val']}
    train_info  = {};
    train_info['train'] = [];
    train_info['val'] = [];

    
    for epoch in range(num_epochs):
        prev_time = time.time()
#        if disp :
#            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#            print('-' * 10)
#
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for batch_i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(inputs,outputs,model)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if disp:
                    #print('{} Loss: {:.4f} '.format(phase, epoch_loss))
                    batches_done = epoch * len(dataloaders[phase]) + batch_i
                    batches_left = num_epochs * len(dataloaders[phase]) - batches_done
                    time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                    prev_time = time.time()
    
    
                    sys.stdout.write(
                        "\r[%s] [Epoch %d/%d] [Batch %d/%d] [Loss: %f] ETA: %s "
                        % (
                            phase,
                            epoch+1,
                            num_epochs,
                            batch_i+1,
                            len(dataloaders[phase]),
                            loss.item(),
                            time_left,
                        )
                    )
    

            epoch_loss = running_loss / dataset_sizes[phase]
            train_info[phase].append(epoch_loss);
            
            if phase == 'train':
                scheduler.step();

    
            if disp:
                print('')
                print('{} Loss: {:.4f} '.format(phase, epoch_loss))           
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        if do_checkpoint>0:
            if epoch%do_checkpoint==0:
                checkpoint(root, epoch, model);

    time_elapsed = time.time() - since
    if disp:
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model , train_info



def train_model_supervised(model, criterion, optimizer, scheduler, dataloaders, device, root, num_epochs=25,disp=False, do_checkpoint=0):
    """ Trains the pytorch model in a supervised way
        """
    since = time.time()
    best_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())

    dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val']}
    train_info  = {};
    train_info['train'] = [];
    train_info['val'] = [];

    
    for epoch in range(num_epochs):
        prev_time = time.time()
#        if disp :
#            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#            print('-' * 10)
#
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for batch_i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(labels,outputs,model)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if disp:
                    #print('{} Loss: {:.4f} '.format(phase, epoch_loss))
                    batches_done = epoch * len(dataloaders[phase]) + batch_i
                    batches_left = num_epochs * len(dataloaders[phase]) - batches_done
                    time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                    prev_time = time.time()
    
    
                    sys.stdout.write(
                        "\r[%s] [Epoch %d/%d] [Batch %d/%d] [Loss: %f] ETA: %s "
                        % (
                            phase,
                            epoch+1,
                            num_epochs,
                            batch_i+1,
                            len(dataloaders[phase]),
                            loss.item(),
                            time_left,
                        )
                    )
    

            epoch_loss = running_loss / dataset_sizes[phase]
            train_info[phase].append(epoch_loss);

            if disp:
                print('')
                print('{} Loss: {:.4f} '.format(phase, epoch_loss))           
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        if do_checkpoint>0:
            if epoch%do_checkpoint==0:
                checkpoint(root, epoch, model);

    time_elapsed = time.time() - since
    if disp:
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model , train_info


class Train_par:
    def __init__(self, batch_size, learning_rate, img_size, reg = 0):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.img_size = img_size;
        self.reg = reg
        self.train_loss = []
        self.val_loss = []
        self.minimum = float("inf")

    def set_loss(self, train_info):
        self.train_loss = train_info['train']
        self.val_loss = train_info['val']
        self.minimum = min(self.val_loss)

    def __str__(self):
        string1 = "Parameters:\nBatch Size : \t {} \nLearning : \t {} \n".format(self.batch_size, self.learning_rate)
        string2 = "Image Size : \t {} \nRegularisation : \t {}".format(self.img_size, self.reg)
        string = string1 + string2;
        return string

    def get_loss(self):
        train_info = {};
        train_info['train'] = self.train_loss;
        train_info['val'] = self.val_loss;
        return train_info;

    def title(self):
        string1 = "Batch_Size_{}_Learning_{}".format(self.batch_size, self.learning_rate)
        string2 = "_size_{}_Regularisation_{}".format(self.img_size, self.reg)
        title = string1 + string2;
        return title



    def plot(self, start = 0):
        plt.ion()
        string1 = "Batch Size : \t {} \ Learning : \t {} \n".format(self.batch_size, self.learning_rate)
        string2 = "size : \t {} \nRegularisation : \t {}".format(self.img_size, self.reg)
        title = string1 + string2;
        
        plt.figure(1 , figsize = (20,10))
        plt.suptitle = title;
        
        Epochs = [i+1 for i in range(start,len(self.train_loss))];

        plt.subplot(2, 1, 1)
        plt.plot(Epochs, self.train_loss[start:], 'o-')
        plt.title('Train')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(2, 1, 2)
        plt.plot(Epochs, self.val_loss[start:], '.-')
        plt.title('Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.show()

    def plot_log(self, start = 0):
        plt.ion()
        string1 = "Batch Size : \t {} \ Learning : \t {} \n".format(self.batch_size, self.learning_rate)
        string2 = "size : \t {} \nRegularisation : \t {}".format(self.img_size, self.reg)
        title = string1 + string2;
        
        plt.figure(1 , figsize = (20,10))
        plt.suptitle = title;
        
        Epochs = [i+1 for i in range(start,len(self.train_loss))];


        train_loss = np.log(self.train_loss);
        val_loss = np.log(self.val_loss);

        plt.subplot(2, 1, 1)
        plt.plot(Epochs, train_loss[start:], 'o-')
        plt.title('Train')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(2, 1, 2)
        plt.plot(Epochs, val_loss[start:], '.-')
        plt.title('Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.show()

def read_param(path):
    with open(path,'rb') as param_file:
        params = pickle.load(param_file);
    return params

######################################################################
# 3. Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
##
# Function to Display reconstruction for a few images
#

def visualize_model(model,dataloaders,device, suptitle="", colormap=plt.cm.gray):
    """
    Takes 8 images from the dataloader and shows side by side the input image and the
    reconstructed image
    """
    plt.ion()   # interactive mode
    inputs, classes = next(iter(dataloaders['train']))
    while inputs.shape[0]<8:
        next_input , classes = next(iter(dataloaders['train']))
        inputs = torch.cat((inputs,next_input),0)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(inputs)

    inputs = inputs.cpu().detach().numpy();
    outputs = outputs.cpu().detach().numpy();
    
    fig, axarr = plt.subplots(4,4,figsize=(20,20));
#    plt.suptitle(suptitle, fontsize = 16)

    for i in range(4):
        for j in range(2):
            im1 = axarr[i,2*j].imshow(inputs[2*i+j,0,:,:],cmap=colormap)
            axarr[i,2*j].set_title("Ground Truth")
            im2 = axarr[i,2*j+1].imshow(outputs[2*i+j,0,:,:], cmap=colormap)
            axarr[i,2*j+1].set_title("Reconstructed")

    plt.subplots_adjust(left = 0.08, wspace = 0.5, top = 0.9, right = 0.9)
    
    
def visualize_conv_layers(conv_layer, suptitle = "", colormap = plt.cm.gray):
    """Displays the 8 first filters of the convolution layer conv_layer
        """
    params = list(conv_layer.parameters());
    conv_filters = params[0];
    conv_filters = conv_filters.cpu().detach().numpy();
    plt.ion()
    (nb_filters, entry_channels, s_x, s_y)=conv_filters.shape;
    
    fig, axarr = plt.subplots(2,4,figsize=(20, 10));
    #plt.suptitle(suptitle, fontsize=16);
    
    for i in range(2):
        for j in range(4):
            nb_pat = 4*i+j;
            if nb_filters>nb_pat:
                Img = conv_filters[nb_pat,0,:,:];
            else:
                Img = np.zeros((s_x, s_y));
            im = axarr[i,j].imshow(Img, cmap = colormap)
            cax = plt.axes([0.02+(j+1)*0.225,0.6-i*0.43,0.005,0.25])
            plt.colorbar(im, cax = cax)

    plt.subplots_adjust(left=0.08, wspace=0.5, top=0.9,right = 0.9);
    plt.show();

######################################################################
# 4. Saving and loading the model so that it can later be utilized
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
def checkpoint(root, epoch, model):
    """ Saves the dictionaries of a given pytorch model for 
        the right epoch
        """
    model_out_path = "model_epoch_{}.pth".format(epoch)
    model_out_path = root / model_out_path;
    torch.save(model.state_dict() , model_out_path);
    print("Checkpoint saved to {}".format(model_out_path))

def save_net(title, model):
    """Saves dictionaries of a given pytorch model in the place defined by 
        title
        """
    model_out_path = "{}.pth".format(title)
    model_out_path = model_out_path;
    torch.save(model.state_dict(), model_out_path);
    print("Model Saved")


def load_net(title, model, device):
    """Loads net defined by title """
    model_out_path = "{}.pth".format(title)
    model.load_state_dict(torch.load(model_out_path, map_location=torch.device(device)))
    print("Model Loaded: {}".format(title))
