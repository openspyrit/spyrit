# -----------------------------------------------------------------------------
#   This software is distributed under the terms
#   of the GNU Lesser General  Public Licence (LGPL)
#   See LICENSE.md for further details
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt;
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image;
import numpy as np;
from numpy import linalg as LA;
import time;
from scipy import signal;
from scipy import misc;
from scipy import sparse
from imutils.video import FPS
import imutils
import cv2
import torch

def display_vid(video, fps, title ='',colormap=plt.cm.gray):
    """
    video is a numpy array of shape [nb_frames, 1, nx, ny]
    """
    plt.ion()
    (nb_frames, channels, nx, ny) = video.shape
    fig = plt.figure();
    ax = fig.add_subplot(1, 1, 1)
    for i in range(nb_frames):
        current_frame = video[i,0,:,:];
        plt.imshow(current_frame, cmap=colormap);
        plt.title(title);
        divider = make_axes_locatable(ax);
        cax = plt.axes([0.85, 0.1, 0.075, 0.8]);
        plt.colorbar(cax=cax);
        plt.show();
        plt.pause(fps)
    plt.ioff()

def display_rgb_vid(video, fps, title =''):
    """
    video is a numpy array of shape [nb_frames, 3, nx, ny]
    """
    plt.ion()
    (nb_frames, channels, nx, ny) = video.shape
    fig = plt.figure();
    ax = fig.add_subplot(1, 1, 1)
    for i in range(nb_frames):
        current_frame = video[i,:,:,:];
        current_frame = np.moveaxis(current_frame, 0, -1)
        plt.imshow(current_frame);
        plt.title(title);
        plt.show();
        plt.pause(fps)
    plt.ioff()


def fitPlots(N, aspect=(16,9)):
    width = aspect[0]
    height = aspect[1]
    area = width*height*1.0
    factor = (N/area)**(1/2.0)
    cols = math.floor(width*factor)
    rows = math.floor(height*factor)
    rowFirst = width < height
    while rows*cols < N:
        if rowFirst:
            rows += 1
        else:
            cols += 1
        rowFirst = not(rowFirst)
    return rows, cols


def Multi_plots(img_list, title_list, shape, suptitle= '', colormap = plt.cm.gray, axis_off = True ,aspect = (16,9), savefig = "", fontsize = 14):
    [rows, cols] = shape;
    plt.figure();
    plt.suptitle(suptitle, fontsize=16);
    if (len(img_list)<rows*cols) or (len(title_list)<rows*cols):
        for k in range(max(rows*cols-len(img_list),rows*cols - len(title_list) )):
            img_list.append(np.zeros((64,64)));
            title_list.append("");

    for k in range(rows*cols):
        ax = plt.subplot(rows,cols,k+1)
        ax.imshow(img_list[k], cmap = colormap)
        ax.set_title(title_list[k], fontsize=fontsize);
        if axis_off :
            plt.axis('off');
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    plt.show()
 

def compare_video_frames(vid_list, nb_disp_frames,title_list, suptitle= '', colormap = plt.cm.gray, aspect = (16,9), savefig = "", fontsize = 14):
    rows = len(vid_list);
    cols = nb_disp_frames;
    plt.figure(figsize=aspect);
    plt.suptitle(suptitle, fontsize=16);
    for i in range(rows):
        for j in range(cols):
            k = (j+1)+(i)*(cols);i
            #print(k)
            ax = plt.subplot(rows,cols,k)
            #print("i = {}, j = {}".format(i,j))
            ax.imshow(vid_list[i][0,j,0,:,:], cmap = colormap)
            ax.set_title(title_list[i][j], fontsize=fontsize);
            plt.axis('off')
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    plt.show()
   
def torch2numpy(torch_tensor):
    return torch_tensor.cpu().detach().numpy();


def uint8(dsp):
    x = (dsp-np.amin(dsp))/(np.amax(dsp)-np.amin(dsp)) * 255;
    x =  x.astype('uint8')
    return x;


def imagesc(Img, title='', colormap=plt.cm.gray):
    """ 
    imagesc(IMG) Display image Img with scaled colors with greyscale 
    colormap and colorbar
    imagesc(IMG, title=ttl) Display image Img with scaled colors with 
    greyscale colormap and colorbar, with the title ttl
    imagesc(IMG, title=ttl, colormap=cmap) Display image Img with scaled colors 
    with colormap and colorbar specified by cmap (choose between 'plasma', 
    'jet', and 'grey'), with the title ttl
    """
    fig = plt.figure();
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(Img, cmap=colormap);
    plt.title(title);
    divider = make_axes_locatable(ax);
    cax = plt.axes([0.85, 0.1, 0.075, 0.8]);
    plt.colorbar(cax=cax);
    plt.show();
    
def imagecomp(Img1, Img2, suptitle='', title1='', title2='', colormap1=plt.cm.gray, colormap2=plt.cm.gray):
    f, (ax1, ax2) = plt.subplots(1, 2)
    im1=ax1.imshow(Img1, cmap=colormap1);
    ax1.set_title(title1);
    cax = plt.axes([0.43, 0.3, 0.025, 0.4]);
    plt.colorbar(im1,cax=cax);
    plt.suptitle(suptitle, fontsize=16);
    #
    im2=ax2.imshow(Img2, cmap=colormap2);
    ax2.set_title(title2);
    cax = plt.axes([0.915, 0.3, 0.025, 0.4]);
    plt.colorbar(im2,cax=cax);
    plt.subplots_adjust(left=0.08, wspace=0.5, top=0.9,right = 0.9);
    plt.show();
    
def imagepanel(Img1, Img2, Img3, Img4, suptitle='', title1='', title2='', title3='',title4='' ,colormap1=plt.cm.gray, colormap2=plt.cm.gray, colormap3=plt.cm.gray, colormap4=plt.cm.gray):
    
    fig, axarr = plt.subplots(2,2,figsize=(20, 10));
    plt.suptitle(suptitle, fontsize=16);
    
    im1=axarr[0,0].imshow(Img1, cmap=colormap1);
    axarr[0, 0].set_title(title1);
    cax = plt.axes([0.4, 0.54, 0.025, 0.35]);
    plt.colorbar(im1,cax=cax);
    
    im2=axarr[0,1].imshow(Img2, cmap=colormap2);
    axarr[0, 1].set_title(title2);
    cax = plt.axes([0.90, 0.54, 0.025, 0.35]);
    plt.colorbar(im2,cax=cax);
    
    im3=axarr[1,0].imshow(Img3, cmap=colormap3);
    axarr[1, 0].set_title(title3);
    cax = plt.axes([0.4, 0.12, 0.025, 0.35]);
    plt.colorbar(im3,cax=cax);
    
    im4=axarr[1,1].imshow(Img4, cmap=colormap4);
    axarr[1, 1].set_title(title4);
    cax = plt.axes([0.9, 0.12, 0.025, 0.35]);
    plt.colorbar(im4,cax=cax);
    
    plt.subplots_adjust(left=0.08, wspace=0.5, top=0.9,right = 0.9);
    plt.show();
    
def plot(x,y,title='',xlabel='', ylabel='', color='black'):
    fig = plt.figure();
    ax = fig.add_subplot(1, 1, 1);
    plt.plot(x, y, color=color);
    plt.title(title);
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    plt.show();
    
def add_colorbar(mappable, position = "right"):
    """
    Example: 
        f, axs = plt.subplots(1, 2)
        im = axs[0].imshow(img1, cmap='gray') 
        add_colorbar(im)
        im = axs[0].imshow(img2, cmap='gray') 
        add_colorbar(im)
    """
    if position=="bottom":
        orientation = 'horizontal'
    else:
        orientation = 'vertical'
    
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax, orientation=orientation)
    plt.sca(last_axes)
    return cbar

def noaxis(axs):
    if type(axs) is np.ndarray:
        for ax in axs:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    else:
        axs.get_xaxis().set_visible(False)
        axs.get_yaxis().set_visible(False)
                
    
def string_mean_std(x, prec = 3):
    return "{:.{p}f} +/- {:.{p}f}".format(np.mean(x), np.std(x), p=prec)
    
def print_mean_std(x, tag='', prec = 3):
    print("{} = {:.{p}f} +/- {:.{p}f}".format(tag, np.mean(x), np.std(x), p=prec)) 
    
def histogram(s):
    count, bins, ignored = plt.hist(s, 30, density=True);
    plt.show();
    
	

def vid2batch(root, img_dim, start_frame, end_frame):
    stream = cv2.VideoCapture(root);
    fps = FPS().start()
    frame_nb = 0;
    output_batch = torch.zeros(1,end_frame-start_frame,1,img_dim, img_dim);
    while True:
        (grabbed, frame) = stream.read()
        if not grabbed:
            break

        frame_nb+=1;
        if (frame_nb>=start_frame)&(frame_nb<end_frame):
            frame = cv2.resize(frame, (img_dim, img_dim));
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            output_batch[0,frame_nb-start_frame,0,:,:] = torch.Tensor(frame[:,:,1]);

    return output_batch;
            
 
def pre_process_video(video, crop_patch, kernel_size):
    batch_size, seq_length, c, h, w = video.shape
    batched_frames = video.view(batch_size*seq_length*c, h, w);
    output_batch = torch.zeros(batched_frames.shape);

    for i in range(batch_size*seq_length*c):
        img = torch2numpy(batched_frames[i,:,:]);
        img[crop_patch] = 0;
        median_frame = cv2.medianBlur(img, kernel_size);
        output_batch[i,:,:] = torch.Tensor(median_frame);
    output_batch = output_batch.view(batch_size, seq_length, c, h, w);
    return output_batch

