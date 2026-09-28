import cmath
import datetime
import glob
import itertools
import math
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import savemat
from scipy.linalg import sqrtm
from numpy import linalg as LA
from sklearn import metrics
from sklearn.decomposition import PCA
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch import Tensor
from torch.autograd import Variable
from torch.backends import cudnn
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
from torchvision.models import vgg19
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.utils import make_grid, save_image
from torchsummary import summary

from torch_geometric.data import DataLoader as PyGDataLoader
from torch_geometric.nn import AvgPool1d, BatchNorm1d, Conv1d, Dropout, ELU, EdgePooling, Linear, MaxPool1d, ReLU, SAGPooling, TopKPooling
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_dense_batch

# ==========================================
# CUDNN & Device Settings
# ==========================================
cudnn.benchmark = False
cudnn.deterministic = True

torch.cuda.set_device(1)
device = torch.device('cuda', 1)

# ==========================================
# Function Definitions
# ==========================================

def generate_cov_matrix(N_r, Horiz_AoA_mean, Vert_AoA_mean):
    beta_k = np.random.uniform(low=1.0, high=1.0, size=None)
    R_k = np.zeros((N_r, N_r), dtype=complex)

    for m in range(N_r):
        for n in range(N_r):
            Horiz_AoA_std = math.pi / 256
            Vert_AoA_std = math.pi / 256
            
            a = 2 * math.pi * Vert_AoA_std * (n - m) * math.cos(Vert_AoA_mean)
            b = 1 + ((Horiz_AoA_std**2) * (a**2) * (math.sin(Horiz_AoA_mean))**2)
            c = 2 * math.pi * (n - m) * math.sin(Vert_AoA_mean)
    
            t_1 = -1 / (2 * b)
            t_2 = (a**2) * ((math.cos(Horiz_AoA_mean))**2)
            t_3 = -2j * c * math.cos(Horiz_AoA_mean)
            t_4 = (Horiz_AoA_std**2) * (c**2) * ((math.sin(Horiz_AoA_mean))**2)

            R_k[m, n] = (beta_k / math.sqrt(b)) * cmath.exp(t_1 * (t_2 + t_3 + t_4)) 
    return R_k


def quantize(Z_data, a_max, t, N_r, bits):
    num = 2**bits
    a_max_repeat = np.repeat(np.expand_dims((a_max), axis=1), t, axis=1)
    Z_arr = np.squeeze(np.array(Z_data))
    m = (np.asarray(divmod(Z_arr + a_max_repeat, a_max_repeat / (num / 2))))[0].astype(int)
    m[m > num - 1] = num - 1
    m[m < 0] = 0
    mapping = (np.transpose(np.linspace(-1 * a_max, a_max, num + 1)) + (np.repeat(np.expand_dims((a_max), axis=1), num + 1, axis=1)) / num)[:, 0:num]
    Y = np.zeros((N_r, t))
    for i in range(N_r):
        for j in range(t):
            Y[i, j] = mapping[i, m[i, j]]
    Y_l = Y - a_max_repeat / num
    Y_U = Y + a_max_repeat / num

    return Y, Y_l, Y_U

# ==========================================
# Hyperparameters & Constants
# ==========================================

t_p = 50
t_d = 450
N_r = 200
sigma_2_w = 10
data_number = 45000
multiple = 3000

divisor = math.sqrt(2)
Constellation = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
M = [symbol / divisor for symbol in Constellation]

# K = 3 Covariance Setup
R_k1 = generate_cov_matrix(N_r, 0, 0)
R_k2 = generate_cov_matrix(N_r, math.pi / 3, math.pi / 3)
R_k3 = generate_cov_matrix(N_r, math.pi / 12, math.pi / 12)

Rk_powerhalf1 = sqrtm(R_k1)
Rk_powerhalf2 = sqrtm(R_k2)
Rk_powerhalf3 = sqrtm(R_k3)

h_mean_temp = np.zeros(N_r)
h_cov_temp = np.identity(N_r) / math.sqrt(2)

h_power_arr = R_k1.diagonal() + R_k2.diagonal() + R_k3.diagonal() 
Recieved_power = np.real(h_power_arr + sigma_2_w)
a_max = 2.5 * np.sqrt(Recieved_power / 2)

Xp_data = []
h_data = []
Xd_data = []
Yd_data = []
Yp_data = []
h_est_data = []

R_real_unnorm = np.zeros((N_r, N_r))
R_img_unnorm = np.zeros((N_r, N_r))

# ==========================================
# Main Data Generation Loop
# ==========================================

index = 0
dict_data = {}
for i in range(20):
    dict_data['data_for_train_{}_center'.format(i)] = {}

for data in range(data_number):
    Y_data = []
    X_data = [] 

    if data % multiple == 0:
        if index > 0:
            dict_data['data_for_train_{}_center'.format(index - 1)]['h_data_gen'] = h_data
            dict_data['data_for_train_{}_center'.format(index - 1)]['Xp_data_gen'] = Xp_data
            dict_data['data_for_train_{}_center'.format(index - 1)]['Xd_data_gen'] = Xd_data
            dict_data['data_for_train_{}_center'.format(index - 1)]['Yd_data_gen'] = Yd_data
            dict_data['data_for_train_{}_center'.format(index - 1)]['Yp_data_gen'] = Yp_data
            dict_data['data_for_train_{}_center'.format(index - 1)]['h_est_data_gen'] = h_est_data

            np.savez('/data/data_for_train_10_center_{}.npz'.format(index - 1), **dict_data['data_for_train_{}_center'.format(index - 1)])

        Xp_data = []
        h_data = []
        Xd_data = []
        Yd_data = []
        Yp_data = []
        h_est_data = []

        index = index + 1

    # Channel Generation
    h1_real_1 = np.random.multivariate_normal(h_mean_temp, h_cov_temp)
    h1_img_1 = np.random.multivariate_normal(h_mean_temp, h_cov_temp)
    h1_temp_1 = h1_real_1 + 1j * h1_img_1
    h1_temp = np.dot(Rk_powerhalf1, h1_temp_1)
    h1_real = np.expand_dims(np.real(h1_temp), axis=1)
    h1_img = np.expand_dims(np.imag(h1_temp), axis=1)
    h1 = np.expand_dims(np.concatenate((h1_real, h1_img), axis=0), axis=0)  

    h2_real_1 = np.random.multivariate_normal(h_mean_temp, h_cov_temp)
    h2_img_1 = np.random.multivariate_normal(h_mean_temp, h_cov_temp)
    h2_temp_1 = h2_real_1 + 1j * h2_img_1
    h2_temp = np.dot(Rk_powerhalf2, h2_temp_1)
    h2_real = np.expand_dims(np.real(h2_temp), axis=1)
    h2_img = np.expand_dims(np.imag(h2_temp), axis=1)
    h2 = np.expand_dims(np.concatenate((h2_real, h2_img), axis=0), axis=0)
    
    h3_real_1 = np.random.multivariate_normal(h_mean_temp, h_cov_temp)
    h3_img_1 = np.random.multivariate_normal(h_mean_temp, h_cov_temp)
    h3_temp_1 = h3_real_1 + 1j * h3_img_1
    h3_temp = np.dot(Rk_powerhalf3, h3_temp_1)
    h3_real = np.expand_dims(np.real(h3_temp), axis=1)
    h3_img = np.expand_dims(np.imag(h3_temp), axis=1)
    h3 = np.expand_dims(np.concatenate((h3_real, h3_img), axis=0), axis=0)

    h = np.concatenate((h1, h2, h3), axis=0) 
    h_data.append(h)

    # Xp, Xd and X Generation
    Xp1_real = np.random.randn(1, t_p)
    Xp1_img = np.random.randn(1, t_p)
    Xd1_temp = np.expand_dims(np.array(random.choices(M, weights=[0.25, 0.25, 0.25, 0.25], k=t_d)), axis=0)
    Xd1_real = np.real(Xd1_temp)
    Xd1_img = np.imag(Xd1_temp)
    X1_real = np.concatenate((Xp1_real, Xd1_real), axis=1)
    X1_img = np.concatenate((Xp1_img, Xd1_img), axis=1)

    X1 = np.expand_dims(np.concatenate((X1_real, X1_img), axis=0), axis=0)
    Xp1 = np.expand_dims(np.concatenate((Xp1_real, Xp1_img), axis=0), axis=0)
    Xd1 = np.expand_dims(np.concatenate((Xd1_real, Xd1_img), axis=0), axis=0)

    Xp2_real = np.random.randn(1, t_p)
    Xp2_img = np.random.randn(1, t_p)
    Xd2_temp = np.expand_dims(np.array(random.choices(M, weights=[0.25, 0.25, 0.25, 0.25], k=t_d)), axis=0)
    Xd2_real = np.real(Xd2_temp)
    Xd2_img = np.imag(Xd2_temp)
    X2_real = np.concatenate((Xp2_real, Xd2_real), axis=1)
    X2_img = np.concatenate((Xp2_img, Xd2_img), axis=1)

    X2 = np.expand_dims(np.concatenate((X2_real, X2_img), axis=0), axis=0)
    Xp2 = np.expand_dims(np.concatenate((Xp2_real, Xp2_img), axis=0), axis=0)
    Xd2 = np.expand_dims(np.concatenate((Xd2_real, Xd2_img), axis=0), axis=0)

    Xp3_real = np.random.randn(1, t_p)
    Xp3_img = np.random.randn(1, t_p)
    Xd3_temp = np.expand_dims(np.array(random.choices(M, weights=[0.25, 0.25, 0.25, 0.25], k=t_d)), axis=0)
    Xd3_real = np.real(Xd3_temp)
    Xd3_img = np.imag(Xd3_temp)
    X3_real = np.concatenate((Xp3_real, Xd3_real), axis=1)
    X3_img = np.concatenate((Xp3_img, Xd3_img), axis=1)

    X3 = np.expand_dims(np.concatenate((X3_real, X3_img), axis=0), axis=0)
    Xp3 = np.expand_dims(np.concatenate((Xp3_real, Xp3_img), axis=0), axis=0)
    Xd3 = np.expand_dims(np.concatenate((Xd3_real, Xd3_img), axis=0), axis=0)

    Xp = np.concatenate((Xp1, Xp2, Xp3), axis=0)
    Xd = np.concatenate((Xd1, Xd2, Xd3), axis=0)    
    X = np.concatenate((X1, X2, X3), axis=0)

    Xp_data.append(Xp)
    Xd_data.append(Xd)
    X_data.append(X)
    
    # Noise Generation
    Wp1_real = np.random.normal(0, math.sqrt(sigma_2_w / 2), (N_r, t_p)) 
    Wp1_imag = np.random.normal(0, math.sqrt(sigma_2_w / 2), (N_r, t_p))
    Wp1 = np.concatenate((Wp1_real, Wp1_imag), axis=0)

    Wd1_real = np.random.normal(0, math.sqrt(sigma_2_w / 2), (N_r, t_d)) 
    Wd1_imag = np.random.normal(0, math.sqrt(sigma_2_w / 2), (N_r, t_d))
    Wd1 = np.concatenate((Wd1_real, Wd1_imag), axis=0) 
   
    # Zp Generation
    Zp1_real = np.dot(h1_real, Xp1_real) - np.dot(h1_img, Xp1_img)  
    Zp1_img = np.dot(h1_img, Xp1_real) + np.dot(h1_real, Xp1_img)
    Zp2_real = np.dot(h2_real, Xp2_real) - np.dot(h2_img, Xp2_img)
    Zp2_img = np.dot(h2_img, Xp2_real) + np.dot(h2_real, Xp2_img) 
    Zp3_real = np.dot(h3_real, Xp3_real) - np.dot(h3_img, Xp3_img) 
    Zp3_img = np.dot(h3_img, Xp3_real) + np.dot(h3_real, Xp3_img) 

    Zp_real = Zp1_real + Zp2_real + Zp3_real + Wp1_real
    Zp_img = Zp1_img + Zp2_img + Zp3_img + Wp1_imag
    Yp_real, Yp_l_real, Yp_U_real = quantize(Zp_real, a_max, t_p, N_r, bits=3)
    Yp_img, Yp_l_img, Yp_U_img = quantize(Zp_img, a_max, t_p, N_r, bits=3)

    Zp_1d = np.expand_dims(np.concatenate((Zp_real, Zp_img), axis=0), axis=0)
    Zp = np.concatenate((Zp_1d, Zp_1d, Zp_1d), axis=0) 
    Yp_1d = np.expand_dims(np.concatenate((Yp_real, Yp_img), axis=0), axis=0)
    Yp = np.concatenate((Yp_1d, Yp_1d, Yp_1d), axis=0)
    Yp_l_1d = np.expand_dims(np.concatenate((Yp_l_real, Yp_l_img), axis=0), axis=0)
    Yp_l = np.concatenate((Yp_l_1d, Yp_l_1d, Yp_l_1d), axis=0)
    Yp_U_1d = np.expand_dims(np.concatenate((Yp_U_real, Yp_U_img), axis=0), axis=0)
    Yp_U = np.concatenate((Yp_U_1d, Yp_U_1d, Yp_U_1d), axis=0)

    Yp_data.append(Yp)

    # Zd Generation
    Zd1_real = np.dot(h1_real, Xd1_real) - np.dot(h1_img, Xd1_img)
    Zd1_img = np.dot(h1_img, Xd1_real) + np.dot(h1_real, Xd1_img)
    Zd2_real = np.dot(h2_real, Xd2_real) - np.dot(h2_img, Xd2_img)
    Zd2_img = np.dot(h2_img, Xd2_real) + np.dot(h2_real, Xd2_img)
    Zd3_real = np.dot(h3_real, Xd3_real) - np.dot(h3_img, Xd3_img)
    Zd3_img = np.dot(h3_img, Xd3_real) + np.dot(h3_real, Xd3_img)

    Zd_real = Zd1_real + Zd2_real + Zd3_real + Wd1_real
    Zd_img = Zd1_img + Zd2_img + Zd3_img + Wd1_imag

    Yd_real, Yd_l_real, Yd_U_real = quantize(Zd_real, a_max, t_d, N_r, bits=3)
    Yd_img, Yd_l_img, Yd_U_img = quantize(Zd_img, a_max, t_d, N_r, bits=3)

    Zd_1d = np.expand_dims(np.concatenate((Zd_real, Zd_img), axis=0), axis=0) 
    Zd = np.concatenate((Zd_1d, Zd_1d, Zd_1d), axis=0) 
    Yd_1d = np.expand_dims(np.concatenate((Yd_real, Yd_img), axis=0), axis=0)
    Yd = np.concatenate((Yd_1d, Yd_1d, Yd_1d), axis=0)
    Yd_l_1d = np.expand_dims(np.concatenate((Yd_l_real, Yd_l_img), axis=0), axis=0)
    Yd_l = np.concatenate((Yd_l_1d, Yd_l_1d, Yd_l_1d), axis=0)
    Yd_U_1d = np.expand_dims(np.concatenate((Yd_U_real, Yd_U_img), axis=0), axis=0)
    Yd_U = np.concatenate((Yd_U_1d, Yd_U_1d, Yd_U_1d), axis=0)

    Yd_data.append(Yd) 

    # Y Concatenation
    Y = np.concatenate((Yp, Yd), axis=2)
    Y_data.append(Y)

    X_data_arr = np.array(X_data, dtype=np.float32)
    Y_data_arr = np.array(Y_data, dtype=np.float32)

# ==========================================
# Final Save Operation
# ==========================================

last_data_dict_index = int((data_number / multiple)) - 1  

dict_data['data_for_train_{}_center'.format(last_data_dict_index)]['h_data_gen'] = h_data
dict_data['data_for_train_{}_center'.format(last_data_dict_index)]['Xp_data_gen'] = Xp_data
dict_data['data_for_train_{}_center'.format(last_data_dict_index)]['Xd_data_gen'] = Xd_data
dict_data['data_for_train_{}_center'.format(last_data_dict_index)]['Yd_data_gen'] = Yd_data
dict_data['data_for_train_{}_center'.format(last_data_dict_index)]['Yp_data_gen'] = Yp_data
dict_data['data_for_train_{}_center'.format(last_data_dict_index)]['h_est_data_gen'] = h_est_data

np.savez('/data/data_for_train_10_center_{}.npz'.format(last_data_dict_index), **dict_data['data_for_train_{}_center'.format(last_data_dict_index)])