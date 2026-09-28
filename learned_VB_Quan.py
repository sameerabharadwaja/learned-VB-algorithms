import math
import cmath
import numpy as np
from scipy.linalg import sqrtm
from numpy import linalg as LA

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# =========================================================================
# HELPER FUNCTIONS & CHANNEL MODEL SETUP
# =========================================================================

def quantize(Z_data, a_max, t, N_r, bits):
    """Quantizes continuous complex/real signals according to bit budget."""
    num = 2**bits
    a_max_repeat = np.repeat(np.expand_dims(a_max, axis=1), t, axis=1)
    Z_arr = np.squeeze(np.array(Z_data))
    
    m = (np.asarray(divmod(Z_arr + a_max_repeat, a_max_repeat / (num / 2))))[0].astype(int)
    m[m > num - 1] = num - 1
    m[m < 0] = 0
    
    mapping = (np.transpose(np.linspace(-1 * a_max, a_max, num + 1)) + 
               (np.repeat(np.expand_dims(a_max, axis=1), num + 1, axis=1)) / num)[:, 0:num]
    
    Y = np.zeros((N_r, t))
    for i in range(N_r):
        for j in range(t):
            Y[i, j] = mapping[i, m[i, j]]
            
    Y_l = Y - a_max_repeat / num
    Y_U = Y + a_max_repeat / num

    return Y, Y_l, Y_U


def generate_cov_matrix(N_r, Horiz_AoA_mean, Vert_AoA_mean):
    """Generates physical spatial covariance matrix R_k based on array geometry."""
    beta_k = 1.0
    R_k = np.zeros((N_r, N_r), dtype=complex)
    Horiz_AoA_std = math.pi / 256
    Vert_AoA_std = math.pi / 256

    for m in range(N_r):
        for n in range(N_r):
            a = 2 * math.pi * Vert_AoA_std * (n - m) * math.cos(Vert_AoA_mean)
            b = 1 + ((Horiz_AoA_std ** 2) * (a ** 2) * (math.sin(Horiz_AoA_mean)) ** 2)
            c = 2 * math.pi * (n - m) * math.sin(Vert_AoA_mean)
            
            t_1 = -1 / (2 * b)
            t_2 = (a ** 2) * ((math.cos(Horiz_AoA_mean)) ** 2)
            t_3 = -2j * c * math.cos(Horiz_AoA_mean)
            t_4 = (Horiz_AoA_std ** 2) * (c ** 2) * ((math.sin(Horiz_AoA_mean)) ** 2)

            R_k[m, n] = (beta_k / math.sqrt(b)) * cmath.exp(t_1 * (t_2 + t_3 + t_4))
            
    return R_k


# =========================================================================
# NEURAL DECODER NETWORK (UNROLLED MIMO CHANNEL ESTIMATOR)
# =========================================================================

class Decoder(nn.Module):
    def __init__(self, sigma_2_w, N_r, t_d, t_p, U1_real, U1_img, E1, U2_real, U2_img, E2, U3_real, U3_img, E3):
        super().__init__()
        
        self.N_r = N_r
        self.sigma_2_w = sigma_2_w
        self.t_p = t_p
        self.t_d = t_d
        
        # Buffer registered spatial properties
        self.register_buffer('linend1_R', U1_real)
        self.register_buffer('linend1_I', U1_img)
        self.register_buffer('delta1', E1)
        
        self.register_buffer('linend2_R', U2_real)
        self.register_buffer('linend2_I', U2_img)
        self.register_buffer('delta2', E2)
        
        self.register_buffer('linend3_R', U3_real)
        self.register_buffer('linend3_I', U3_img)
        self.register_buffer('delta3', E3)

        self.linear1 = nn.Linear(4 * N_r, 32)
        self.linear2 = nn.Linear(32, 28)
        self.linear3 = nn.Linear(28, 2 * N_r)
        self.relu = nn.ReLU()

    def forward(self, zp, xp, h1, batch_size, Yp):
        # 1. Structural Covariance Computation
        U_Real_1 = torch.unsqueeze(self.linend1_R, dim=0)
        U_Img_1 = torch.unsqueeze(self.linend1_I, dim=0)
        U_Real_2 = torch.unsqueeze(self.linend2_R, dim=0)
        U_Img_2 = torch.unsqueeze(self.linend2_I, dim=0)
        U_Real_3 = torch.unsqueeze(self.linend3_R, dim=0)
        U_Img_3 = torch.unsqueeze(self.linend3_I, dim=0)

        U_Real = torch.cat((U_Real_1, U_Real_2, U_Real_3), dim=0)
        U_Img = torch.cat((U_Img_1, U_Img_2, U_Img_3), dim=0)

        delta1_ = torch.squeeze(self.delta1)
        delta2_ = torch.squeeze(self.delta2)
        delta3_ = torch.squeeze(self.delta3)
        
        delta_inv_1 = 1.0 / delta1_
        delta_inv_2 = 1.0 / delta2_
        delta_inv_3 = 1.0 / delta3_
        
        Diag = torch.zeros(batch_size, 3, self.N_r, self.N_r, device=xp.device, dtype=xp.dtype)

        # Batch-wide Norm Calculation & Diagonal Matrix Construction
        for i in range(batch_size):
            temp_0 = 1.0 / (((torch.square(torch.linalg.norm(xp[i, 0])) + self.t_d) / self.sigma_2_w) + delta_inv_1)
            temp_1 = 1.0 / (((torch.square(torch.linalg.norm(xp[i, 1])) + self.t_d) / self.sigma_2_w) + delta_inv_2)
            temp_2 = 1.0 / (((torch.square(torch.linalg.norm(xp[i, 2])) + self.t_d) / self.sigma_2_w) + delta_inv_3)
            
            Diag[i, 0] = torch.diag(temp_0)
            Diag[i, 1] = torch.diag(temp_1)
            Diag[i, 2] = torch.diag(temp_2)

        learn_h_cov_R = torch.matmul(torch.matmul(U_Real, Diag), torch.permute(U_Real, (0, 2, 1))) + \
                        torch.matmul(torch.matmul(U_Img, Diag), torch.permute(U_Img, (0, 2, 1)))
                        
        learn_h_cov_I = torch.matmul(torch.matmul(U_Img, Diag), torch.permute(U_Real, (0, 2, 1))) - \
                        torch.matmul(torch.matmul(U_Real, Diag), torch.permute(U_Img, (0, 2, 1)))

        # 2. Successive Interference Cancellation & Real/Imaginary Split
        h1_real = h1[:, :, 0:self.N_r, :]
        h1_img = h1[:, :, self.N_r:2 * self.N_r, :]

        zp_real = zp[:, :, 0:self.N_r, :]
        zp_img = zp[:, :, self.N_r:2 * self.N_r, :]

        xp_real = xp[:, :, 0:1, :]
        xp_img = xp[:, :, 1:2, :]

        xp_summation_real = torch.matmul(h1_real, xp_real) - torch.matmul(h1_img, xp_img)
        xp_summation_img = torch.matmul(h1_real, xp_img) + torch.matmul(h1_img, xp_real)

        xp_totalsum_real = torch.unsqueeze(torch.sum(xp_summation_real, dim=1), dim=1)
        xp_totalsum_img = torch.unsqueeze(torch.sum(xp_summation_img, dim=1), dim=1)

        xp_term2_summation_real = torch.cat((xp_totalsum_real, xp_totalsum_real, xp_totalsum_real), dim=1)
        xp_term2_summation_img = torch.cat((xp_totalsum_img, xp_totalsum_img, xp_totalsum_img), dim=1)

        zp_term2_real = zp_real - (xp_term2_summation_real - xp_summation_real)
        zp_term2_img = zp_img - (xp_term2_summation_img - xp_summation_img)

        h1_A_real = torch.matmul(zp_term2_real, torch.permute(xp_real, (0, 1, 3, 2))) + \
                     torch.matmul(zp_term2_img, torch.permute(xp_img, (0, 1, 3, 2)))
        h1_A_img = torch.matmul(zp_term2_img, torch.permute(xp_real, (0, 1, 3, 2))) - \
                    torch.matmul(zp_term2_real, torch.permute(xp_img, (0, 1, 3, 2)))

        S_R = h1_A_real
        S_I = h1_A_img

        h1_real = (1.0 / self.sigma_2_w) * (torch.matmul(learn_h_cov_R, S_R) - torch.matmul(learn_h_cov_I, S_I))
        h1_img = (1.0 / self.sigma_2_w) * (torch.matmul(learn_h_cov_R, S_I) + torch.matmul(learn_h_cov_I, S_R))

        h1 = torch.cat((h1_real, h1_img), dim=2)

        z = torch.matmul(U_Real, torch.permute(U_Real, (0, 2, 1))) + torch.matmul(U_Img, torch.permute(U_Img, (0, 2, 1)))
        
        Diag_delta1 = torch.unsqueeze(torch.diag(delta1_), dim=0)
        Diag_delta2 = torch.unsqueeze(torch.diag(delta2_), dim=0)
        Diag_delta3 = torch.unsqueeze(torch.diag(delta3_), dim=0)
        Diag_delta = torch.cat((Diag_delta1, Diag_delta2, Diag_delta3), dim=0)

        Learned_R_Real = torch.matmul(torch.matmul(U_Real, Diag_delta), torch.permute(U_Real, (0, 2, 1))) + \
                          torch.matmul(torch.matmul(U_Img, Diag_delta), torch.permute(U_Img, (0, 2, 1)))
        Learned_R_Img = torch.matmul(torch.matmul(U_Img, Diag_delta), torch.permute(U_Real, (0, 2, 1))) - \
                         torch.matmul(torch.matmul(U_Real, Diag_delta), torch.permute(U_Img, (0, 2, 1)))

        Learned_R = torch.cat((Learned_R_Real, Learned_R_Img), dim=1)

        # 3. Dense Neural Refining Layer Forward Pass
        exp_zp1_real_1 = torch.matmul(h1_real, xp_real) - torch.matmul(h1_img, xp_img)
        exp_zp1_img_1 = torch.matmul(h1_img, xp_real) + torch.matmul(h1_real, xp_img)

        exp_zp1_real = torch.sum(exp_zp1_real_1, dim=1)
        exp_zp1_img = torch.sum(exp_zp1_img_1, dim=1)

        Yp_real = Yp[:, 0, 0:self.N_r, :]
        Yp_img = Yp[:, 0, self.N_r:2 * self.N_r, :]

        exp_p = torch.cat((exp_zp1_real, exp_zp1_img, Yp_real, Yp_img), dim=1)
        exp_p_d_T = torch.permute(exp_p, (0, 2, 1))

        x1 = self.relu(self.linear1(exp_p_d_T))
        x1 = self.relu(self.linear2(x1))
        x1 = self.relu(self.linear3(x1))

        x1_T = torch.permute(x1, (0, 2, 1))

        exp_zp_1 = x1_T[:, :, 0:self.t_p]
        exp_zp_1 = torch.unsqueeze(exp_zp_1, dim=1)
        exp_zp = torch.cat((exp_zp_1, exp_zp_1, exp_zp_1), dim=1)

        return h1, z, Learned_R, exp_zp


# =========================================================================
# MAIN EXECUTION & OPTIMIZATION PIPELINE
# =========================================================================

if __name__ == "__main__":
    t_p = 500
    t_d = 0
    N_r = 10
    sigma_2_w_ls = [20, 30]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for sigma_2_w in sigma_2_w_ls:
        print(f"\n================ Running for sigma_2_w = {sigma_2_w} ================")
        
        # 1. Spatial Covariance Structure Generation
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

        Xp_data, h_data, Yp_data = [], [], []

        # 2. Dataset Synthesis Loop
        print("Generating synthetic MIMO channels and quantized signals...")
        for _ in range(16384):
            # --- Channel Generation ---
            h1_temp_1 = np.random.multivariate_normal(h_mean_temp, h_cov_temp) + \
                        1j * np.random.multivariate_normal(h_mean_temp, h_cov_temp)
            h1_temp = np.dot(Rk_powerhalf1, h1_temp_1)
            h1 = np.expand_dims(np.concatenate((np.expand_dims(np.real(h1_temp), axis=1), 
                                                np.expand_dims(np.imag(h1_temp), axis=1)), axis=0), axis=0)

            h2_temp_1 = np.random.multivariate_normal(h_mean_temp, h_cov_temp) + \
                        1j * np.random.multivariate_normal(h_mean_temp, h_cov_temp)
            h2_temp = np.dot(Rk_powerhalf2, h2_temp_1)
            h2 = np.expand_dims(np.concatenate((np.expand_dims(np.real(h2_temp), axis=1), 
                                                np.expand_dims(np.imag(h2_temp), axis=1)), axis=0), axis=0)

            h3_temp_1 = np.random.multivariate_normal(h_mean_temp, h_cov_temp) + \
                        1j * np.random.multivariate_normal(h_mean_temp, h_cov_temp)
            h3_temp = np.dot(Rk_powerhalf3, h3_temp_1)
            h3 = np.expand_dims(np.concatenate((np.expand_dims(np.real(h3_temp), axis=1), 
                                                np.expand_dims(np.imag(h3_temp), axis=1)), axis=0), axis=0)

            h = np.concatenate((h1, h2, h3), axis=0)
            h_data.append(h)

            # --- Pilot Generation ---
            Xp1 = np.expand_dims(np.concatenate((np.random.randn(1, t_p), np.random.randn(1, t_p)), axis=0), axis=0)
            Xp2 = np.expand_dims(np.concatenate((np.random.randn(1, t_p), np.random.randn(1, t_p)), axis=0), axis=0)
            Xp3 = np.expand_dims(np.concatenate((np.random.randn(1, t_p), np.random.randn(1, t_p)), axis=0), axis=0)
            Xp = np.concatenate((Xp1, Xp2, Xp3), axis=0)
            Xp_data.append(Xp)

            # --- Signal Reception & ADC Quantization ---
            Wp1_real = np.random.normal(0, math.sqrt(sigma_2_w / 2), (N_r, t_p))
            Wp1_imag = np.random.normal(0, math.sqrt(sigma_2_w / 2), (N_r, t_p))

            Zp1_real = np.dot(h1[0, :N_r, :], Xp1[0, 0:1, :]) - np.dot(h1[0, N_r:, :], Xp1[0, 1:2, :])
            Zp1_img = np.dot(h1[0, N_r:, :], Xp1[0, 0:1, :]) + np.dot(h1[0, :N_r, :], Xp1[0, 1:2, :])

            Zp2_real = np.dot(h2[0, :N_r, :], Xp2[0, 0:1, :]) - np.dot(h2[0, N_r:, :], Xp2[0, 1:2, :])
            Zp2_img = np.dot(h2[0, N_r:, :], Xp2[0, 0:1, :]) + np.dot(h2[0, :N_r, :], Xp2[0, 1:2, :])

            Zp3_real = np.dot(h3[0, :N_r, :], Xp3[0, 0:1, :]) - np.dot(h3[0, N_r:, :], Xp3[0, 1:2, :])
            Zp3_img = np.dot(h3[0, N_r:, :], Xp3[0, 0:1, :]) + np.dot(h3[0, :N_r, :], Xp3[0, 1:2, :])

            Zp_real = Zp1_real + Zp2_real + Zp3_real + Wp1_real
            Zp_img = Zp1_img + Zp2_img + Zp3_img + Wp1_imag

            Yp_real, _, _ = quantize(Zp_real, a_max, t_p, N_r, bits=2)
            Yp_img, _, _ = quantize(Zp_img, a_max, t_p, N_r, bits=2)

            Yp_1d = np.expand_dims(np.concatenate((Yp_real, Yp_img), axis=0), axis=0)
            Yp = np.concatenate((Yp_1d, Yp_1d, Yp_1d), axis=0)
            Yp_data.append(Yp)

        Xp_data_arr = torch.from_numpy(np.array(Xp_data, dtype=np.float32))
        h_data_arr = torch.from_numpy(np.array(h_data, dtype=np.float32))
        Yp_data_arr = torch.from_numpy(np.array(Yp_data, dtype=np.float32))

        batch_size = 512
        n_epochs = 31
        lr = 0.0005
        num_layer = 20

        # --- Eigendecomposition ---
        def extract_eigen_tensors(R):
            vals, vecs = LA.eig(R)
            vals_expand = torch.from_numpy(np.real(np.expand_dims(vals, axis=1)).astype('float32')).to(device)
            vecs_R = torch.from_numpy(np.real(vecs).astype('float32')).to(device)
            vecs_I = torch.from_numpy(np.imag(vecs).astype('float32')).to(device)
            return vecs_R, vecs_I, vals_expand

        U1_real, U1_img, E1 = extract_eigen_tensors(R_k1)
        U2_real, U2_img, E2 = extract_eigen_tensors(R_k2)
        U3_real, U3_img, E3 = extract_eigen_tensors(R_k3)

        # 3. Model Initialization & Pre-trained Weights Loading
        model = Decoder(sigma_2_w, N_r, t_d, t_p, U1_real, U1_img, E1, U2_real, U2_img, E2, U3_real, U3_img, E3).to(device)

        ep = 50
        checkpoint_path = f'/data/2_NMSE_MIMO_{sigma_2_w}_epoch_{ep}.pth'
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            with torch.no_grad():
                model.linear1.weight.copy_(checkpoint['linear1.weight'].to(device))
                model.linear1.bias.copy_(checkpoint['linear1.bias'].to(device))
                model.linear2.weight.copy_(checkpoint['linear2.weight'].to(device))
                model.linear2.bias.copy_(checkpoint['linear2.bias'].to(device))
                model.linear3.weight.copy_(checkpoint['linear3.weight'].to(device))
                model.linear3.bias.copy_(checkpoint['linear3.bias'].to(device))
            print(f"Successfully loaded warm-start weights from {checkpoint_path}")
        except FileNotFoundError:
            print(f"Warning: Checkpoint not found at {checkpoint_path}. Training from scratch.")

        dataset = TensorDataset(Xp_data_arr, h_data_arr, Yp_data_arr)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Pre-allocate zero-initialization tensors to prevent inner-loop memory allocations
        h_init_zeros = torch.zeros((batch_size, 3, 1, 2 * N_r), device=device, dtype=torch.float32)

        # 4. Neural Unrolling Optimization Loop
        for e in range(n_epochs):
            total_loss = 0.0
            model.train()

            for i, (Xp_batch, h_batch, Yp_batch) in enumerate(dataloader):
                Xp_batch = Xp_batch.to(device)
                h_batch = h_batch.to(device)
                Yp_batch = Yp_batch.to(device)

                h1 = h_init_zeros.clone()
                Zp_batch = Yp_batch.clone()

                # Recurrent unrolling over physical network layers
                for j in range(num_layer):
                    h_out, _, _, exp_Zp = model(Zp_batch, Xp_batch, h1, batch_size, Yp_batch)
                    h1 = h_out
                    Zp_batch = exp_Zp

                h_label_flat = torch.reshape(h_batch[:, :, :, 0], (batch_size * 3, -1))
                h_out_flat = torch.reshape(h_out[:, :, :, 0], (batch_size * 3, -1))

                # Normalized Mean Squared Error (NMSE) Loss Computation
                norm_diff = torch.linalg.norm(h_label_flat - h_out_flat, dim=1)
                norm_label = torch.linalg.norm(h_label_flat, dim=1)
                loss = torch.mean(torch.square(norm_diff / norm_label))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f'Epoch [{e:02d}/{n_epochs}] -> Normalized Train NMSE: {avg_loss:.6f}')

            if (e + 1) % 10 == 0:
                save_path = f'/data/3_NMSE_MIMO_{sigma_2_w}_epoch_{e+1}.pth'
                torch.save(model.state_dict(), save_path)
                print(f"Saved epoch checkpoint to {save_path}")