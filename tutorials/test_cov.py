# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 18:53:39 2021

@author: ducros
"""


#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


#%%
#- Acquisition
img_size = 64 # image size
#batch_size = 1024

#- Model and data paths
data_root = Path('./data/')
stat_root = Path('./stats_test/')

#- Save plot using type 1 font
plt.rcParams['pdf.fonttype'] = 42


#%%
from spyrit.misc.statistics import stat_walsh_stl10, Cov2Var
stat_walsh_stl10(stat_root)


#%% Comparison with an existing covariance matrix
print('Loading Cov') 
Cov = np.load(stat_root / Path("Cov_{}x{}.npy".format(img_size, img_size)))
stat_root = Path('./stats_walsh/')
Cov_ref = np.load(stat_root / Path("Cov_{}x{}.npy".format(img_size, img_size)))

Var_ref = Cov2Var(Cov_ref)
Var = Cov2Var(Cov)
err = Var_ref - Var

err_cov = np.linalg.norm(Cov_ref-Cov)/np.linalg.norm(Cov_ref)
err_var = np.linalg.norm(Var_ref-Var)/np.linalg.norm(Var_ref)

print(f"Cov error: {err_cov}")
print(f"Var error: {err_var}")

f, axs = plt.subplots(1,2)
axs[0].imshow(np.divide(Var_ref,Var), cmap='gray')
axs[0].set_title(f"Var ratio: ref / computed")
axs[1].imshow(Var_ref-Var, cmap='gray'); 
axs[1].set_title(f"Var diff: ref - computed")

f, axs = plt.subplots(1,2)
axs[0].imshow(np.divide(Cov_ref, Cov), cmap='gray')
axs[0].set_title(f"Cov ratio: ref / computed")
axs[1].imshow(Cov_ref-Cov, cmap='gray'); 
axs[1].set_title(f"Cov diff: ref - computed")

#%% Comparison of variance masks
from spyrit.learning.model_Had_DCAN import Variance_mask

eta1 = .5
eta2 = .25
mask1 = Variance_mask(Cov_ref,eta1)
mask2 = Variance_mask(Cov_ref,eta2)

f, axs = plt.subplots(1, 3, figsize=(12,8),  dpi= 100)
f.suptitle('ref')
axs[0].imshow(np.log(Var_ref), cmap='gray'); 
axs[1].imshow(mask1, cmap='gray');
axs[2].imshow(mask2, cmap='gray');
axs[0].set_title("variance")
axs[1].set_title(f"mask {eta1}")
axs[2].set_title(f"mask {eta2}")

eta1 = .5
eta2 = .25
mask1 = Variance_mask(Cov,eta1)
mask2 = Variance_mask(Cov,eta2)

f, axs = plt.subplots(1, 3, figsize=(12,8),  dpi= 100)
f.suptitle('computed')
axs[0].imshow(np.log(Var), cmap='gray'); 
axs[1].imshow(mask1, cmap='gray');
axs[2].imshow(mask2, cmap='gray');
axs[0].set_title("variance")
axs[1].set_title(f"mask {eta1}")
axs[2].set_title(f"mask {eta2}")