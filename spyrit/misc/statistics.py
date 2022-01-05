from __future__ import print_function, division
import torch
import torchvision
#from torchvision import datasets, transforms
from pathlib import Path
from spyrit.learning.model_Had_DCAN import *
import time
import spyrit.misc.walsh_hadamard as wh
import numpy as np


def stat_walsh(dataloader, device, root):
    
    # Get dimensions and estimate total number of images in the dataset
    inputs, classes = next(iter(dataloader))
    (b, c, nx, ny) = inputs.shape
    tot_num = len(dataloader)*b
    
    # 1. Mean
    
    # Init
    n = 0
    mean = torch.zeros((nx,ny), dtype=torch.float32)
    H = wh.walsh_matrix(nx).astype(np.float32, copy=False)
    
    # Send to device (e.g., cuda)
    mean = mean.to(device)
    H = torch.from_numpy(H).to(device)
    
    # Accumulate sum over all images in dataset
    for inputs,_ in dataloader:
        inputs = inputs.to(device);
        trans = wh.walsh2_torch(inputs,H)
        mean = mean.add(torch.sum(trans,0))
        # print
        n = n + inputs.shape[0]
        print(f'Mean:  {n} / (less than) {tot_num} images', end='\n')
    print('', end='\n')
    
    # Normalize
    mean = mean/n
    mean = torch.squeeze(mean)
    #torch.save(mean, root+'Average_{}x{}'.format(nx,ny)+'.pth')

    path = root / Path('Average_{}x{}'.format(nx,ny)+'.npy')
    if not root.exists():
        root.mkdir()
    np.save(path, mean.cpu().detach().numpy())
    
    # 2. Covariance
    
    # Init
    n = 0
    cov = torch.zeros((nx*ny,nx*ny), dtype=torch.float32)
    cov = cov.to(device)
    
    # Accumulate (im - mu)*(im - mu)^T over all images in dataset
    for inputs,_ in dataloader:
        inputs = inputs.to(device)
        trans = wh.walsh2_torch(inputs,H)
        trans = trans - mean.repeat(inputs.shape[0],1,1,1)
        trans = trans.view(inputs.shape[0], nx*ny, 1)
        cov = torch.addbmm(cov, trans, trans.view(inputs.shape[0], 1, nx*ny))
        # print
        n += inputs.shape[0]
        print(f'Cov:  {n} / (less than) {tot_num} images', end='\n')
    print('', end='\n')
    
    # Normalize
    cov = cov/(n-1)
    #torch.save(cov, root+'Cov_{}x{}'.format(nx,ny)+'.pth') # todo?

    path = root / Path('Cov_{}x{}'.format(nx,ny)+'.npy')
    if not root.exists():
        root.mkdir()
    np.save(path, cov.cpu().detach().numpy())
    
    return mean, cov

def stat_walsh_np(dataloader, root):
    """ 
        Computes Mean Hadamard Image over the whole dataset + 
        Covariance Matrix Amongst the coefficients
    """
    inputs, classes = next(iter(dataloader))
    inputs = inputs.cpu().detach().numpy()
    (batch_size, channels, nx, ny) = inputs.shape
    tot_num = len(dataloader)*batch_size
    
    H1d = wh.walsh_ordered(nx)
    
     # Abs matrix
    Mean_had = abs_walsh_ordered(dataloader, H1d, tot_num)
    print("Saving abs")
    np.save(root / Path('Abs_{}x{}'.format(nx,ny)+'.npy'), Mean_had)

    # Mean matrix
    #-- Accumulate over all images in dataset
    n = 0
    Mean_had = np.zeros((nx, ny))
    for inputs,_ in dataloader:
        inputs = inputs.cpu().detach().numpy()
        for i in range(inputs.shape[0]):
            img = inputs[i,0,:,:]
            h_img = wh.walsh_ordered2(img,H1d)
            Mean_had += h_img
            n = n+1
        print(f'Mean:  {n} / (less than) {tot_num} images', end='\r')
    print('', end='\n')
    
    #-- Normalize & save
    Mean_had = Mean_had/n;
    print("Saving mean")
    np.save(root / Path('Mean_{}x{}'.format(nx,ny)+'.npy'), Mean_had)
    
    # Covariance matrix    
    n = 0
    Cov_had = np.zeros((nx*ny, nx*ny));
    for inputs,_ in dataloader:
        inputs = inputs.cpu().detach().numpy();
        for i in range(inputs.shape[0]):
            img = inputs[i,0,:,:];
            h_img = wh.walsh_ordered2(img, H1d);
            Norm_Variable = np.reshape(h_img-Mean_had, (nx*ny,1));
            Cov_had += Norm_Variable*np.transpose(Norm_Variable);
            n = n+1
        print(f'Covariance:  {n} / (less than) {tot_num} images', end='\r')     
    print()
    
    #-- Normalize & save
    Cov_had = Cov_had/(n-1);  
    np.save(root / Path('Cov_{}x{}'.format(nx,ny)+'.npy'), Cov_had)

def abs_walsh(dataloader, device):
    
    # Estimate tot_num
    inputs, classes = next(iter(dataloader))
    #inputs = inputs.cpu().detach().numpy();
    (batch_size, channels, nx, ny) = inputs.shape;
    tot_num = len(dataloader)*batch_size;
    
    # Init
    n = 0
    output = torch.zeros((nx,ny),dtype=torch.float32)
    H = wh.walsh_matrix(nx).astype(np.float32, copy=False)
    
    # Send to device (e.g., cuda)
    output = output.to(device)
    H = torch.from_numpy(H).to(device)
    
    # Accumulate over all images in dataset
    for inputs,_ in dataloader:
        inputs = inputs.to(device);
        n = n + inputs.shape[0]
        trans = wh.walsh2_torch(inputs,H);
        trans = torch.abs(trans)
        output = output.add(torch.sum(trans,0))
        print(f'Abs:  {n} / (less than) {tot_num} images', end='\n')
    print('', end='\n')
    
    #-- Normalize
    output = output/n;
    output = torch.squeeze(output)
    
    return output

def Stat_had(dataloader, root):
    """ 
        Computes Mean Hadamard Image over the whole dataset + 
        Covariance Matrix Amongst the coefficients
    """

    inputs, classes = next(iter(dataloader))
    inputs = inputs.cpu().detach().numpy();
    (batch_size, channels, nx, ny) = inputs.shape;
    tot_num = len(dataloader)*batch_size;

    Mean_had = np.zeros((nx, ny));
    for inputs,labels in dataloader:
        inputs = inputs.cpu().detach().numpy();
        for i in range(inputs.shape[0]):
            img = inputs[i,0,:,:];
            H = wh.walsh_matrix(len(img))
            h_img = wh.walsh2(img,H)/len(img)
            Mean_had += h_img;
    Mean_had = Mean_had/tot_num;

    Cov_had = np.zeros((nx*ny, nx*ny));
    for inputs,labels in dataloader:
        inputs = inputs.cpu().detach().numpy();
        for i in range(inputs.shape[0]):
            img = inputs[i,0,:,:];
            H = wh.walsh_matrix(len(img))
            h_img = wh.walsh2(img,H)/len(img)
            Norm_Variable = np.reshape(h_img-Mean_had, (nx*ny,1));
            Cov_had += Norm_Variable*np.transpose(Norm_Variable);
    Cov_had = Cov_had/(tot_num-1);

    np.save(root+'Cov_{}x{}'.format(nx,ny)+'.npy', Cov_had)
    np.savetxt(root+'Cov_{}x{}'.format(nx,ny)+'.txt', Cov_had)
    
    np.save(root+'Average_{}x{}'.format(nx,ny)+'.npy', Mean_had)
    np.savetxt(root+'Average_{}x{}'.format(nx,ny)+'.txt', Mean_had)
    cv2.imwrite(root+'Average_{}x{}'.format(nx,ny)+'.png', Mean_had) #Needs conversion to Uint8!
    return Mean_had, Cov_had 

def optim_had(dataloader, root):
    """ Computes image that ranks the hadamard coefficients
    """
    inputs, classes = next(iter(dataloader))
    inputs = inputs.cpu().detach().numpy();
    (batch_size, channels, nx, ny) = inputs.shape;

    tot_num = len(dataloader)*batch_size;
    Cumulated_had = np.zeros((nx, ny));
    # Iterate over data.
    for inputs,labels in dataloader:
        inputs = inputs.cpu().detach().numpy();
        for i in range(inputs.shape[0]):
            img = inputs[i,0,:,:];
            H = wh.walsh_matrix(len(img))
            h_img = wh.walsh2(img,H)/len(img)
            h_img = np.abs(h_img)/tot_num;
            Cumulated_had += h_img;
    
    Cumulated_had = Cumulated_had / np.max(Cumulated_had) * 255
    np.save(root+'{}x{}'.format(nx,ny)+'.npy', Cumulated_had)
    np.savetxt(root+'{}x{}'.format(nx,ny)+'.txt', Cumulated_had)
    cv2.imwrite(root+'{}x{}'.format(nx,ny)+'.png', Cumulated_had)
    return Cumulated_had 

def Cov2Var(Cov):
    """
    Extracts Variance Matrix from Covarience Matrix
    """
    (Nx, Ny) = Cov.shape;
    diag_index = np.diag_indices(Nx);
    Var = Cov[diag_index];
    Var = np.reshape(Var, (int(np.sqrt(Nx)),int(np.sqrt(Nx))) );
    return Var

def stat_walsh_stl10(stat_root = Path('./stats/'), data_root = Path('./data/'),
                    img_size = 64, batch_size = 1024):


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7) # for reproductibility

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.functional.to_grayscale,
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5])])

    trainset = \
        torchvision.datasets.STL10(root=data_root, split='train+unlabeled',download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)

    testset = \
        torchvision.datasets.STL10(root=data_root, split='test',download=True, transform=transform)
    testloader =  torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False)

    dataloaders = {'train':trainloader, 'val':testloader}

    # Walsh ordered transforms
    time_start = time.perf_counter()
    stat_walsh(dataloaders['train'], device, stat_root)
    time_elapsed = (time.perf_counter() - time_start)
    print(time_elapsed)
    
def stat_mean_coef_from_model(dataloader, device, model_exp):
    #A rediscuter avec Nicolas
    # Get dimensions and estimate total number of images in the dataset
    inputs, classes = next(iter(dataloader))
    (ny,nh) = mdt.size()
 
    mean = torch.zeros(nh).to(device)
#   
    for inputs,_ in dataloader:
        inputs = inputs.to(device);
        trans = torch.matmul(inputs,model_exp)#.cpu() 
        mean = mean.add(torch.sum(trans,[0,1]))
    mean_vect = np.abs(np.transpose(mean.cpu().detach().numpy())) 
    #mean = mean/mean.max()
    return(mean_vect)

def mea_abs_model(dataloader, device, model,root):
    
    # Get dimensions and estimate total number of images in the dataset
    inputs, classes = next(iter(dataloader))
    (nx,nh) = model.size()
    tot_num = len(dataloader)
    (b,ny,nx) = inputs.size()
    # 1. Mean
    
    # Init
    n = 0
    mean = torch.zeros(nh).to(device)
    
    # Accumulate sum over all images in dataset
    for inputs,_ in dataloader:
        inputs = inputs.to(device);
        trans = torch.abs(torch.matmul(inputs,model))#.cpu() 
        mean = mean.add(torch.sum(trans,[0,1]))
        # print
        n += inputs.shape[0]
       # print(f'Mean:  {n} / (less than) {tot_num} images', end='\n')
    print('', end='\n')
    
    # Normalize
    mean = mean/(n*ny)
    #print(mean.size())
    mean = torch.squeeze(mean)
    #torch.save(mean, root+'Average_{}x{}'.format(nx,ny)+'.pth')

    path = root / Path('W_Average_abs_Nx{}_Nh{}'.format(nx,nh)+'.npy')
    #if not root.exists():
    #    root.mkdir()
    np.save(path, mean.cpu().detach().numpy())
    
    return(mean)
    
def stat_model(dataloader, device, model,root):
    
    # Get dimensions and estimate total number of images in the dataset
    inputs, classes = next(iter(dataloader))
    (nx,nh) = model.size()
    tot_num = len(dataloader)
    (b,ny,nx) = inputs.size()
    # 1. Mean
    
    # Init
    n = 0
    mean = torch.zeros(nh).to(device)
    
    # Accumulate sum over all images in dataset
    for inputs,_ in dataloader:
        inputs = inputs.to(device);
        trans = torch.matmul(inputs,model)#.cpu() 
        mean = mean.add(torch.sum(trans,[0,1]))
        # print
        n += inputs.shape[0]
       # print(f'Mean:  {n} / (less than) {tot_num} images', end='\n')
    print('', end='\n')
    
    # Normalize
    mean = mean/(n*ny)
    #print(mean.size())
    mean = torch.squeeze(mean)
    #torch.save(mean, root+'Average_{}x{}'.format(nx,ny)+'.pth')

    path = root / Path('W_Average_Nx{}_Nh{}'.format(nx,nh)+'.npy')
    #if not root.exists():
    #    root.mkdir()
    np.save(path, mean.cpu().detach().numpy())
    
    # 2. Covariance
    
    # Init
    n = 0
    cov = torch.zeros((nh,nh), dtype=torch.float32)
    cov = cov.to(device)
    
    # Accumulate (im - mu)*(im - mu)^T over all images in dataset
    for inputs,_ in dataloader:
        inputs = inputs.to(device)
        trans = torch.matmul(inputs,model)
        #print(trans.size())
        for i in range(ny):
            im_mu = trans[0,i] - mean

            cov += torch.matmul(im_mu.reshape(nh,1),im_mu.reshape(1,nh))
        # print
        n += inputs.shape[0]
        #print(f'Cov:  {n} / (less than) {tot_num} images', end='\n')
    print('', end='\n')
    
    # Normalize
    cov = cov/((n*ny)-1)
    #torch.save(cov, root+'Cov_{}x{}'.format(nx,ny)+'.pth') # todo?

    path = root / Path('W_Cov_Nx{}_Nh{}'.format(nx,nh)+'.npy')
    #if not root.exists():
    #    root.mkdir()
    np.save(path, cov.cpu().detach().numpy())
    
    return mean, cov


