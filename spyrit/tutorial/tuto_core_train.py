r"""
04. Tutorial 2D - Tutorial to train a reconstruction network 
======================
This tutorial shows how to train a reconstruction network for 2D single-pixel imaging 
on stl10. Training is performed by a call to *train.py*. Several parameters allow 
to modify acquisition, network and training (network architecture), 
optimisation and the use of tensorboard. 
 
Currently you can train the following networks by modifying the network architecture variable *arch*: 
 
    - 'dc-net': Denoised Completion Network (DCNet). 
    - 'pinv-net': Pseudo Inverse Network (PinvNet).
    - 'upgd': Unrolled proximal gradient descent (UPGD). 
    
and the denoising variable *denoi*: E
    - 'cnn': CNN no batch normalization
    - 'cnnbn': CNN with batch normalization
    - 'unet': UNet (0.5 M trainable parameters) 


[Colab version]: https://colab.research.google.com/github/openspyrit/spyrit/blob/master/spyrit/tutorial/tuto_train_colab.ipynb

"""

###############################################################################
# Import packages
import os
import datetime
import subprocess

###############################################################################
# Download covariance matrix. Alternatively install *openspyrit/spas* package:
# 
#     spyrit
#     ├───stat
#     │   ├───Average_64x64.npy
#     │   ├───Cov_64x64.npy
#     ├───spirit
# 

download_cov = True
if (download_cov is True):
    # pip install girder-client
    import girder_client

    # api Rest url of the warehouse
    url='https://pilot-warehouse.creatis.insa-lyon.fr/api/v1'
    
    # Generate the warehouse client
    gc = girder_client.GirderClient(apiUrl=url)

    # Download the covariance matrix and mean image
    data_folder = './stat/'
    dataId_list = [
            '63935b624d15dd536f0484a5', # for reconstruction (imageNet, 64)
            '63935a224d15dd536f048496', # for reconstruction (imageNet, 64)
            ]
    for dataId in dataId_list:
        myfile = gc.getFile(dataId)
        gc.downloadFile(dataId, data_folder + myfile['name'])

    print(f'Created {data_folder}') 
    os.listdir(data_folder)

###############################################################################
r""" Train
You can choose the following parameters:
 
    - Acquisition: 
        - --img_size: Height / width dimension, default=64
        - --M: Number of undersampling patterns, default=512
        - --subs: Among 'var','rect', default="var"
     
    - Network and training: 
        - --data: stl10 or imagenet, default="stl10"
        - --model_root: Path to model saving files, default='./model/'
        - --data_root: Path to the dataset, default="./data/"
    
        - --N0: Mean maximum total number of photons, default=10
        - --stat_root: Path to precomputed data, default="./stat/"
        - --arch: Choose among 'dc-net','pinv-net', 'upgd', default="dc-net"
        - --denoi: Choose among 'cnn','cnnbn', 'unet', default="unet"
        - --device", Choose among 'cuda','cpu', default="" (cuda if available)

    - Optimisation:
        - --num_epochs: Number of training epochs, default=30
        - --batch_size: Size of each training batch, default=512
        - --reg: Regularisation Parameter, default=1e-7
        - --step_size: Scheduler Step Size, default=10
        - --gamma: Scheduler Decrease Rate, default=0.5
        - --checkpoint_model: Optional path to checkpoint model, default=""
        - --checkpoint_interval: Interval between saving model checkpoints, default=0
        - Training is done with *Adam* optimizer, *MSELoss*
 
    - Tensorboard:
        - --tb_path: Relative path for Tensorboard experiment tracking logs, default=False
        - --tb_prof: Code profiler with Tensorboard, default=False
        - Logging of scalars *train_loss*, *val_loss* and images (dataset example ground-truth and predictions at different epochs).
"""

###############################################################################
# In this tutorial, data is perturbed by Poisson noise (100 mean photons) 
# and undersampling factor of 4, on stl10 dataset.
# Training is done with default parameters (see above) and using experiment tracking with tensorboard. 

# Parameters
N0 = 100
M = 1024
data_root = './data/'
data = 'stl10'
stat_root = './stat'

# Tensorboard logs path
now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
tb_path = f'runs/runs_stdl10_n100_m1024/{now}' 
tb_prof = True # False

# Run train.py
cmd = ['python', 'spyrit/tutorial/train.py', '--N0', str(N0), '--M', str(M), '--data_root', str(data_root), 
       '--data', str(data), '--stat_root', str(stat_root), '--tb_path', str(tb_path), '--tb_prof', str(tb_prof), 
       '--device', str('cpu')]
subprocess.run(cmd, check=True)

###############################################################################
# Tensorboard
#
# To launch tensorboard, run in a terminal:  
# tensorboard --logdir <tb_path>
#
# Select *SCALARS* or *IMAGES*. More options are available in the top-right corner.

