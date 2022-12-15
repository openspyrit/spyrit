from __future__ import print_function, division
import torch
import torchvision
#from torchvision import datasets, transforms
from pathlib import Path
#from spyrit.learning.model_Had_DCAN import *
import time
import spyrit.misc.walsh_hadamard as wh
import numpy as np
from scipy.stats import rankdata

def stat_walsh_ImageNet(stat_root = Path('./stats/'), 
                        data_root = Path('./data/ILSVRC2012_img_test_v10102019/'),
                        img_size = 128, 
                        batch_size = 256, 
                        n_loop=1,
                        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                        ):
    """ 
    Args:
        'data_root' needs to have all images in a subfolder
    
    Example:
        from pathlib import Path
        from spyrit.misc.statistics import stat_walsh_ImageNet
        
        data_root =  Path('../data/ILSVRC2012_v10102019')
        stat_root =  Path('../stat/ILSVRC2012_v10102019')
        stat_walsh_ImageNet(stat_root = stat_root, data_root = data_root,
                            img_size = 32, batch_size = 1024)
           
    """

    dataloaders = data_loaders_ImageNet(data_root, img_size=img_size, 
                                        batch_size=batch_size, seed=7)

    # Walsh ordered transforms
    time_start = time.perf_counter()
    stat_walsh(dataloaders['train'], device, stat_root, n_loop)
    time_elapsed = (time.perf_counter() - time_start)
    
    print(f'Computed in {time_elapsed} seconds')

def stat_walsh_stl10(stat_root = Path('./stats/'), 
                     data_root = Path('./data/'),
                     img_size = 64, 
                     batch_size = 1024,
                     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                     ):
    """
    Args:
        'data_root' is expected to contain an 'stl10_binary' subfolder with the 
        test*.bin, train*.bin and unlabeled_X.bin files.
    
    Example:
        data_root =  Path('../datasets/')
        stat_root =  Path('../stat/stl10')
    
        from spyrit.misc.statistics import stat_walsh_stl10
        stat_walsh_stl10(stat_root = stat_root, data_root = data_root)
    
    """
    dataloaders = data_loaders_stl10(data_root, img_size=img_size, 
                                     batch_size=batch_size, seed=7)
    # Walsh ordered transforms
    time_start = time.perf_counter()
    stat_walsh(dataloaders['train'], device, stat_root)
    time_elapsed = (time.perf_counter() - time_start)
    
    print(f'Computed in {time_elapsed} seconds')
    
def data_loaders_ImageNet(train_root, val_root=None, img_size=64, 
                          batch_size=512, seed=7, shuffle=False): 
    """ 
    Args:
        Both 'train_root' and 'val_root' need to have images in a subfolder
        shuffle=True to shuffle train set only (test set not shuffled)
        
    The output of torchvision datasets are PILImage images in the range [0, 1].
    We transform them to Tensors in the range [-1, 1]. Also RGB images are 
    converted into grayscale images.   
    """

    torch.manual_seed(seed) # reproductibility of random crop
    #    
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.functional.to_grayscale,
         torchvision.transforms.RandomCrop(
             size=(img_size, img_size), pad_if_needed=True, padding_mode='edge'),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize([0.5], [0.5])
        ])
    
    # train set
    trainset = torchvision.datasets.ImageFolder(root=train_root, transform=transform)
    trainloader =  torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
    
    # validation set (if any)
    if val_root is not None:
        valset = torchvision.datasets.ImageFolder(root=val_root, transform=transform)
        valloader =  torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
    else:
        valloader = None
        
    dataloaders = {'train':trainloader, 'val':valloader}
    
    return dataloaders

def data_loaders_stl10(data_root, img_size=64, batch_size=512, seed=7, 
                       shuffle=False): 
    """ 
    Args:
        shuffle=True to shuffle train set only (test set not shuffled)
        
    The output of torchvision datasets are PILImage images in the range [0, 1].
    We transform them to Tensors in the range [-1, 1]. Also RGB images are 
    converted into grayscale images.
        
    """
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.functional.to_grayscale,
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.STL10(root=data_root, split='train+unlabeled',
                                          download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=shuffle)
 
    testset = torchvision.datasets.STL10(root=data_root, split='test',
                                         download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)
    
    dataloaders = {'train':trainloader, 'val':testloader}
    
    return dataloaders

def stat_walsh(dataloader, device, root, n_loop=1):
    """ 
    nloop > 1 is relevant for dataloaders with random crops such as that 
    provided by data_loaders_ImageNet
        
    """
    # Get dimensions and estimate total number of images in the dataset
    inputs, _ = next(iter(dataloader))
    (_, _, nx, ny) = inputs.shape

    #--------------------------------------------------------------------------
    # 1. Mean
    #--------------------------------------------------------------------------
    mean = mean_walsh(dataloader, device, n_loop=n_loop)
    
    # Save
    if n_loop==1:
        path = root / Path('Average_{}x{}'.format(nx,ny)+'.npy')
    else:
        path = root / Path('Average_{}_{}x{}'.format(n_loop,nx,ny)+'.npy')
        
    if not root.exists():
        root.mkdir()
    np.save(path, mean.cpu().detach().numpy())
    #--------------------------------------------------------------------------
    # 2. Covariance
    #-------------------------------------------------------------------------
    cov = cov_walsh(dataloader, mean, device, n_loop=n_loop)
        
    # Save
    if n_loop==1:
        path = root / Path('Cov_{}x{}'.format(nx,ny)+'.npy')
    else:
        path = root / Path('Cov_{}_{}x{}'.format(n_loop,nx,ny)+'.npy')
        
    if not root.exists():
        root.mkdir()
    np.save(path, cov.cpu().detach().numpy())
    
    return mean, cov

def mean_walsh(dataloader, device, n_loop=1):
    """ 
    nloop > 1 is relevant for dataloaders with random crops such as that 
    provided by data_loaders_ImageNet
        
    """
    
    # Get dimensions and estimate total number of images in the dataset
    inputs, _ = next(iter(dataloader))
    (b, c, nx, ny) = inputs.shape
    tot_num = len(dataloader)*b
    
    # Init
    n = 0
    H = wh.walsh_matrix(nx).astype(np.float32, copy=False)
    mean = torch.zeros((nx,ny), dtype=torch.float32)
    
    # Send to device (e.g., cuda)
    mean = mean.to(device)
    H = torch.from_numpy(H).to(device)
    
    # Compute Mean 
    # Accumulate sum over all images in dataset
    for i in range(n_loop):
        torch.manual_seed(i)
        for inputs,_ in dataloader:
            inputs = inputs.to(device)
            trans = wh.walsh2_torch(inputs,H)
            mean = mean.add(torch.sum(trans,0))
            # print
            n = n + inputs.shape[0]
            print(f'Mean:  {n} / (less than) {tot_num*n_loop} images', end='\n')
            # test
            #print(f' | {inputs[53,0,33,49]}', end='\n')
        print('', end='\n')
    
    # Normalize
    mean = mean/n
    mean = torch.squeeze(mean)
    
    return mean   
    
def cov_walsh(dataloader, mean, device, n_loop=1):
    """ 
    nloop > 1 is relevant for dataloaders with random crops such as that 
    provided by data_loaders_ImageNet
        
    """
    
    # Get dimensions and estimate total number of images in the dataset
    inputs, _ = next(iter(dataloader))
    (b, c, nx, ny) = inputs.shape
    tot_num = len(dataloader)*b
    
    H = wh.walsh_matrix(nx).astype(np.float32, copy=False)
    H = torch.from_numpy(H).to(device)
    
    # Covariance --------------------------------------------------------------
    # Init
    n = 0
    cov = torch.zeros((nx*ny,nx*ny), dtype=torch.float32)
    cov = cov.to(device)
    
    # Accumulate (im - mu)*(im - mu)^T over all images in dataset
    for i in range(n_loop):
        torch.manual_seed(i)
        for inputs,_ in dataloader:
            inputs = inputs.to(device)
            trans = wh.walsh2_torch(inputs,H)
            trans = trans - mean.repeat(inputs.shape[0],1,1,1)
            trans = trans.view(inputs.shape[0], nx*ny, 1)
            cov = torch.addbmm(cov, trans, trans.view(inputs.shape[0], 1, nx*ny))
            # print
            n += inputs.shape[0]
            print(f'Cov:  {n} / (less than) {tot_num*n_loop} images', end='\n')
            # test
            #print(f' | {inputs[53,0,33,49]}', end='\n')
        print('', end='\n')
    
    # Normalize
    cov = cov/(n-1)
    
    return cov

def stat_fwalsh_S(dataloader, device, root): # NOT validated!
    
    # Get dimensions and estimate total number of images in the dataset
    inputs, classes = next(iter(dataloader))
    (b, c, nx, ny) = inputs.shape
    tot_num = len(dataloader)*b
    
    # 1. Mean
    
    # Init
    n = 0
    mean = torch.zeros((nx,ny), dtype=torch.float32)
    
    # Send to device (e.g., cuda)
    mean = mean.to(device)
    ind = wh.sequency_perm_ind(nx*ny)
    
    # Accumulate sum over all images in dataset
    for inputs,_ in dataloader:
        inputs = inputs.to(device);
        trans = wh.fwalsh2_S_torch(inputs,ind)
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
        trans = wh.fwalsh2_S_torch(inputs,ind)
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

def Cov2Var(Cov):
    """
    Extracts Variance Matrix from Covariance Matrix
    """
    (Nx, Ny) = Cov.shape;
    diag_index = np.diag_indices(Nx);
    Var = Cov[diag_index];
    Var = np.reshape(Var, (int(np.sqrt(Nx)),int(np.sqrt(Nx))) );
    return Var


def img2mask(Ord, M):
    """
    Returns subsampling mask from order matrix
    """
    (nx, ny) = Ord.shape;
    msk = np.ones((nx, ny));
    ranked_data = np.reshape(rankdata(-Ord, method = 'ordinal'),(nx, ny));
    msk[np.absolute(ranked_data)>M]=0;
    return msk


# todo: rewrite in a fashion similar to stat_walsh_stl10
def stat_fwalsh_S_stl10(stat_root = Path('./stats/'), data_root = Path('./data/'),
                    img_size = 64, batch_size = 1024):
    
    """Fast Walsh S-transform of X in "2D"

    Args:
        :attr:`X` (torch.tensor):  input image with shape `(*, n, n)`. `n`**2 
                                    should be a power of two.


    Returns:
        torch.tensor: S-transformed signal with shape `(*, n, n)`
    
    Examples:
        >>> import spyrit.misc.statistics as st
        >>> st.stat_fwalsh_S_stl10()
        
    """ 


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
    stat_fwalsh_S(dataloaders['train'], device, stat_root)
    time_elapsed = (time.perf_counter() - time_start)
    print(time_elapsed)

#%% delete ? Deprecated ?

# def stat_walsh_np(dataloader, root):
#     """ 
#         Computes Mean Hadamard Image over the whole dataset + 
#         Covariance Matrix Amongst the coefficients
#     """
#     inputs, classes = next(iter(dataloader))
#     inputs = inputs.cpu().detach().numpy()
#     (batch_size, channels, nx, ny) = inputs.shape
#     tot_num = len(dataloader)*batch_size
    
#     H1d = wh.walsh_ordered(nx)
    
#      # Abs matrix
#     Mean_had = abs_walsh_ordered(dataloader, H1d, tot_num)
#     print("Saving abs")
#     np.save(root / Path('Abs_{}x{}'.format(nx,ny)+'.npy'), Mean_had)

#     # Mean matrix
#     #-- Accumulate over all images in dataset
#     n = 0
#     Mean_had = np.zeros((nx, ny))
#     for inputs,_ in dataloader:
#         inputs = inputs.cpu().detach().numpy()
#         for i in range(inputs.shape[0]):
#             img = inputs[i,0,:,:]
#             h_img = wh.walsh_ordered2(img,H1d)
#             Mean_had += h_img
#             n = n+1
#         print(f'Mean:  {n} / (less than) {tot_num} images', end='\r')
#     print('', end='\n')
    
#     #-- Normalize & save
#     Mean_had = Mean_had/n;
#     print("Saving mean")
#     np.save(root / Path('Mean_{}x{}'.format(nx,ny)+'.npy'), Mean_had)
    
#     # Covariance matrix    
#     n = 0
#     Cov_had = np.zeros((nx*ny, nx*ny));
#     for inputs,_ in dataloader:
#         inputs = inputs.cpu().detach().numpy();
#         for i in range(inputs.shape[0]):
#             img = inputs[i,0,:,:];
#             h_img = wh.walsh_ordered2(img, H1d);
#             Norm_Variable = np.reshape(h_img-Mean_had, (nx*ny,1));
#             Cov_had += Norm_Variable*np.transpose(Norm_Variable);
#             n = n+1
#         print(f'Covariance:  {n} / (less than) {tot_num} images', end='\r')     
#     print()
    
#     #-- Normalize & save
#     Cov_had = Cov_had/(n-1);  
#     np.save(root / Path('Cov_{}x{}'.format(nx,ny)+'.npy'), Cov_had)

#%% delete ? Deprecated ?

# def abs_walsh(dataloader, device):
    
#     # Estimate tot_num
#     inputs, classes = next(iter(dataloader))
#     #inputs = inputs.cpu().detach().numpy();
#     (batch_size, channels, nx, ny) = inputs.shape;
#     tot_num = len(dataloader)*batch_size;
    
#     # Init
#     n = 0
#     output = torch.zeros((nx,ny),dtype=torch.float32)
#     H = wh.walsh_matrix(nx).astype(np.float32, copy=False)
    
#     # Send to device (e.g., cuda)
#     output = output.to(device)
#     H = torch.from_numpy(H).to(device)
    
#     # Accumulate over all images in dataset
#     for inputs,_ in dataloader:
#         inputs = inputs.to(device);
#         n = n + inputs.shape[0]
#         trans = wh.walsh2_torch(inputs,H);
#         trans = torch.abs(trans)
#         output = output.add(torch.sum(trans,0))
#         print(f'Abs:  {n} / (less than) {tot_num} images', end='\n')
#     print('', end='\n')
    
#     #-- Normalize
#     output = output/n;
#     output = torch.squeeze(output)
    
#     return output

#%% delete ? Deprecated ?

# def Stat_had(dataloader, root):
#     """ 
#         Computes Mean Hadamard Image over the whole dataset + 
#         Covariance Matrix Amongst the coefficients
#     """

#     inputs, classes = next(iter(dataloader))
#     inputs = inputs.cpu().detach().numpy();
#     (batch_size, channels, nx, ny) = inputs.shape;
#     tot_num = len(dataloader)*batch_size;

#     Mean_had = np.zeros((nx, ny));
#     for inputs,labels in dataloader:
#         inputs = inputs.cpu().detach().numpy();
#         for i in range(inputs.shape[0]):
#             img = inputs[i,0,:,:];
#             H = wh.walsh_matrix(len(img))
#             h_img = wh.walsh2(img,H)/len(img)
#             Mean_had += h_img;
#     Mean_had = Mean_had/tot_num;

#     Cov_had = np.zeros((nx*ny, nx*ny));
#     for inputs,labels in dataloader:
#         inputs = inputs.cpu().detach().numpy();
#         for i in range(inputs.shape[0]):
#             img = inputs[i,0,:,:];
#             H = wh.walsh_matrix(len(img))
#             h_img = wh.walsh2(img,H)/len(img)
#             Norm_Variable = np.reshape(h_img-Mean_had, (nx*ny,1));
#             Cov_had += Norm_Variable*np.transpose(Norm_Variable);
#     Cov_had = Cov_had/(tot_num-1);

#     np.save(root+'Cov_{}x{}'.format(nx,ny)+'.npy', Cov_had)
#     np.savetxt(root+'Cov_{}x{}'.format(nx,ny)+'.txt', Cov_had)
    
#     np.save(root+'Average_{}x{}'.format(nx,ny)+'.npy', Mean_had)
#     np.savetxt(root+'Average_{}x{}'.format(nx,ny)+'.txt', Mean_had)
#     cv2.imwrite(root+'Average_{}x{}'.format(nx,ny)+'.png', Mean_had) #Needs conversion to Uint8!
#     return Mean_had, Cov_had 

#%% Keep or delete?
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
    
#%% What for? Still in use? Ask Antonio?
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
