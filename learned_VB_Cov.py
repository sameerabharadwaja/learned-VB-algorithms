import math
import cmath
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# 1. Device Configuration
# ==========================================
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(2)


# ==========================================
# 2. Network Components & Submodules
# ==========================================
class Decoder(nn.Module):
    def __init__(self, all_ones, P_s_init, S0, S1, S2, S3, sigma_2_w, N_r, t_d, t_p):
        super().__init__()
        self.N_r = N_r
        self.t_d = t_d
        self.t_p = t_p
        self.sigma_2_w = sigma_2_w

        self.all_ones = all_ones
        self.P_s_init = P_s_init
        self.S0, self.S1, self.S2, self.S3 = S0, S1, S2, S3

        # Linear transformations for covariance orthogonal factorizations
        self.linend1_R = nn.Linear(N_r, N_r)
        self.linend1_I = nn.Linear(N_r, N_r)
        self.delta1 = nn.Linear(1, N_r)

        self.linend2_R = nn.Linear(N_r, N_r)
        self.linend2_I = nn.Linear(N_r, N_r)
        self.delta2 = nn.Linear(1, N_r)

        self.linend3_R = nn.Linear(N_r, N_r)
        self.linend3_I = nn.Linear(N_r, N_r)
        self.delta3 = nn.Linear(1, N_r)

    def forward(self, zd, zp, xp, h1, E_real, E_img, batch_size):
        # --- A. Symbol Energy & Norm Estimation ---
        term_1 = torch.zeros(batch_size, 3, 1, self.t_d, device=device)
        for j in range(batch_size):
            term_1[j, 0, :, :] = torch.square(torch.norm(h1[j, 0, :self.N_r, 0])) * self.all_ones[j, 0, :, :]
            term_1[j, 1, :, :] = torch.square(torch.norm(h1[j, 1, :self.N_r, 0])) * self.all_ones[j, 1, :, :]
            term_1[j, 2, :, :] = torch.square(torch.norm(h1[j, 2, :self.N_r, 0])) * self.all_ones[j, 2, :, :]

        zd_real = zd[:, :, :self.N_r, :]
        zd_img  = zd[:, :, self.N_r:2*self.N_r, :]

        S0_real, S0_imag = self.S0[:, :self.t_d, :], self.S0[:, self.t_d:2*self.t_d, :]
        S1_real, S1_imag = self.S1[:, :self.t_d, :], self.S1[:, self.t_d:2*self.t_d, :]
        S2_real, S2_imag = self.S2[:, :self.t_d, :], self.S2[:, self.t_d:2*self.t_d, :]
        S3_real, S3_imag = self.S3[:, :self.t_d, :], self.S3[:, self.t_d:2*self.t_d, :]

        h1_real = h1[:, :, :self.N_r, :]
        h1_img  = h1[:, :, self.N_r:2*self.N_r, :]
        
        h1_real_T = torch.permute(h1_real, (0, 1, 3, 2))
        h1_img_T  = torch.permute(h1_img, (0, 1, 3, 2))

        # --- B. Interference Cancellation & Detection ---
        xd_sum_real = torch.matmul(h1_real, E_real) - torch.matmul(h1_img, E_img)
        xd_sum_img  = torch.matmul(h1_real, E_img)  + torch.matmul(h1_img, E_real)

        totalsum_real = xd_sum_real.sum(dim=1, keepdim=True)
        totalsum_img  = xd_sum_img.sum(dim=1, keepdim=True)

        term2_real = zd_real - (totalsum_real.repeat(1, 3, 1, 1) - xd_sum_real)
        term2_img  = zd_img  - (totalsum_img.repeat(1, 3, 1, 1)  - xd_sum_img)

        # Log-Likelihood Updates for QPSK Symbols
        def compute_F(S_real, S_imag):
            FA = torch.matmul(term2_real, S_real) + torch.matmul(term2_img, S_imag)
            FB = torch.matmul(term2_img, S_real)  - torch.matmul(term2_real, S_imag)
            return (-1.0 / self.sigma_2_w) * (term_1 - 2.0 * (torch.matmul(h1_real_T, FA) + torch.matmul(h1_img_T, FB))) + self.P_s_init

        F_s0 = compute_F(S0_real, S0_imag)
        F_s1 = compute_F(S1_real, S1_imag)
        F_s2 = compute_F(S2_real, S2_imag)
        F_s3 = compute_F(S3_real, S3_imag)

        F_concat = torch.cat((F_s0, F_s1, F_s2, F_s3), dim=2)
        F_probs = F.softmax(F_concat, dim=2)

        inv_sqrt2 = 1.0 / math.sqrt(2)
        E_real_out = (inv_sqrt2 * (F_probs[:, :, 0, :] - F_probs[:, :, 1, :] - F_probs[:, :, 2, :] + F_probs[:, :, 3, :])).unsqueeze(2)
        E_img_out  = (inv_sqrt2 * (F_probs[:, :, 0, :] + F_probs[:, :, 1, :] - F_probs[:, :, 2, :] - F_probs[:, :, 3, :])).unsqueeze(2)

        # --- C. Channel Covariance & Eigenstructure Estimation ---
        I_mat = torch.eye(self.N_r, device=device)

        def normalize_U(lin_R, lin_I):
            u_r = lin_R(I_mat)
            u_i = lin_I(I_mat)
            norm_val = torch.sqrt(u_r.norm(dim=0, keepdim=True)**2 + u_i.norm(dim=0, keepdim=True)**2)
            return (u_r / norm_val).unsqueeze(0), (u_i / norm_val).unsqueeze(0)

        U_r1, U_i1 = normalize_U(self.linend1_R, self.linend1_I)
        U_r2, U_i2 = normalize_U(self.linend2_R, self.linend2_I)
        U_r3, U_i3 = normalize_U(self.linend3_R, self.linend3_I)

        U_Real = torch.cat((U_r1, U_r2, U_r3), dim=0)
        U_Img  = torch.cat((U_i1, U_i2, U_i3), dim=0)

        l_vec = torch.ones(1, 1, device=device)
        delta1_ = self.delta1(l_vec).squeeze()
        delta2_ = self.delta2(l_vec).squeeze()
        delta3_ = self.delta3(l_vec).squeeze()

        Diag = torch.zeros(batch_size, 3, self.N_r, self.N_r, device=device)
        for i in range(batch_size):
            Diag[i, 0] = torch.diag(1.0 / (((xp[i, 0].norm()**2 + self.t_d) / self.sigma_2_w) + (1.0 / delta1_)))
            Diag[i, 1] = torch.diag(1.0 / (((xp[i, 1].norm()**2 + self.t_d) / self.sigma_2_w) + (1.0 / delta2_)))
            Diag[i, 2] = torch.diag(1.0 / (((xp[i, 2].norm()**2 + self.t_d) / self.sigma_2_w) + (1.0 / delta3_)))

        learn_h_cov_R = torch.matmul(torch.matmul(U_Real, Diag), U_Real.permute(0, 2, 1)) + torch.matmul(torch.matmul(U_Img, Diag), U_Img.permute(0, 2, 1))
        learn_h_cov_I = torch.matmul(torch.matmul(U_Img, Diag), U_Real.permute(0, 2, 1)) - torch.matmul(torch.matmul(U_Real, Diag), U_Img.permute(0, 2, 1))

        # --- D. Channel Refinement Updates ---
        zp_real = zp[:, :, :self.N_r, :]
        zp_img  = zp[:, :, self.N_r:2*self.N_r, :]
        xp_real = xp[:, :, 0:1, :]
        xp_img  = xp[:, :, 1:2, :]

        xp_sum_real = torch.matmul(h1_real, xp_real) - torch.matmul(h1_img, xp_img)
        xp_sum_img  = torch.matmul(h1_real, xp_img)  + torch.matmul(h1_img, xp_real)

        xp_totalsum_real = xp_sum_real.sum(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        xp_totalsum_img  = xp_sum_img.sum(dim=1, keepdim=True).repeat(1, 3, 1, 1)

        zp_term2_real = zp_real - (xp_totalsum_real - xp_sum_real)
        zp_term2_img  = zp_img  - (xp_totalsum_img  - xp_sum_img)

        h1_A_real = torch.matmul(zp_term2_real, xp_real.permute(0, 1, 3, 2)) + torch.matmul(zp_term2_img, xp_img.permute(0, 1, 3, 2))
        h1_A_img  = torch.matmul(zp_term2_img,  xp_real.permute(0, 1, 3, 2)) - torch.matmul(zp_term2_real, xp_img.permute(0, 1, 3, 2))

        h1_B_real = torch.matmul(term2_real, E_real.permute(0, 1, 3, 2)) + torch.matmul(term2_img, E_img.permute(0, 1, 3, 2))
        h1_B_img  = torch.matmul(term2_img,  E_real.permute(0, 1, 3, 2)) - torch.matmul(term2_real, E_img.permute(0, 1, 3, 2))

        S_R = h1_A_real + h1_B_real
        S_I = h1_A_img  + h1_B_img

        h1_real_out = (1.0 / self.sigma_2_w) * (torch.matmul(learn_h_cov_R, S_R) - torch.matmul(learn_h_cov_I, S_I))
        h1_img_out  = (1.0 / self.sigma_2_w) * (torch.matmul(learn_h_cov_R, S_I) + torch.matmul(learn_h_cov_I, S_R))

        h1_out = torch.cat((h1_real_out, h1_img_out), dim=2)
        z = torch.matmul(U_Real, U_Real.permute(0, 2, 1)) + torch.matmul(U_Img, U_Img.permute(0, 2, 1))

        Diag_delta = torch.stack((torch.diag(delta1_), torch.diag(delta2_), torch.diag(delta3_)), dim=0)
        Learned_R_Real = torch.matmul(torch.matmul(U_Real, Diag_delta), U_Real.permute(0, 2, 1)) + torch.matmul(torch.matmul(U_Img, Diag_delta), U_Img.permute(0, 2, 1))
        Learned_R_Img  = torch.matmul(torch.matmul(U_Img, Diag_delta), U_Real.permute(0, 2, 1)) - torch.matmul(torch.matmul(U_Real, Diag_delta), U_Img.permute(0, 2, 1))

        Learned_R = torch.cat((Learned_R_Real, Learned_R_Img), dim=1)

        return F_concat, h1_out, z, Learned_R, E_real_out, E_img_out


# ==========================================
# 3. Main Unfolding Model (Stacked Layers)
# ==========================================
class DeepUnfoldingModel(nn.Module):
    def __init__(self, num_layers, all_ones, P_s_init, S0, S1, S2, S3, sigma_2_w, N_r, t_d, t_p):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            Decoder(all_ones, P_s_init, S0, S1, S2, S3, sigma_2_w, N_r, t_d, t_p)
            for _ in range(num_layers)
        ])

    def forward(self, zd, zp, xp, h1_init, E_real_init, E_img_init, batch_size):
        h1, E_real, E_img = h1_init, E_real_init, E_img_init
        
        for layer in self.layers:
            F_concat, h1, z, Learned_R, E_real, E_img = layer(zd, zp, xp, h1, E_real, E_img, batch_size)

        return F_concat, h1, z, Learned_R, E_real, E_img


# ==========================================
# 4. Training Setup & Execution
# ==========================================
def train_epoch(model, dataloader, optimizer, criterion_ce, criterion_mse, R_real, N_r, device):
    model.train()
    total_epoch_loss = 0.0

    for batch_idx, (zd, zp, xp, h_true, Xd_real, Xd_img) in enumerate(dataloader):
        batch_size = zd.size(0)

        # Send data to device
        zd = zd.to(device)
        zp = zp.to(device)
        xp = xp.to(device)
        h_true = h_true.to(device)
        Xd_real = Xd_real.to(device)
        Xd_img = Xd_img.to(device)

        # Initial conditions / zero states
        h1_init = torch.zeros(batch_size, 3, 2 * N_r, 1, device=device)
        E_real_init = torch.zeros(batch_size, 3, 1, zd.size(3), device=device)
        E_img_init  = torch.zeros(batch_size, 3, 1, zd.size(3), device=device)

        optimizer.zero_grad()

        # Forward pass through unfolded layers
        F_concat, h1_pred, z, Learned_R, E_real_pred, E_img_pred = model(
            zd, zp, xp, h1_init, E_real_init, E_img_init, batch_size
        )

        # --- A. Data Detection Loss (Cross-Entropy) ---
        # Vectorized target decision generation for QPSK symbols
        real_flat = Xd_real.squeeze(-1)
        imag_flat = Xd_img.squeeze(-1)

        target_cls = torch.zeros_like(real_flat, dtype=torch.long)
        target_cls[(real_flat < 0) & (imag_flat > 0)] = 1
        target_cls[(real_flat < 0) & (imag_flat < 0)] = 2
        target_cls[(real_flat > 0) & (imag_flat < 0)] = 3

        train_targets = F.one_hot(target_cls, num_classes=4).float().to(device)
        
        # Softmax classification loss across detection outputs
        F_probs = F.softmax(F_concat, dim=2)
        loss1 = criterion_ce(F_probs, train_targets)

        # --- B. Channel Estimation Loss (MSE) ---
        loss2 = criterion_mse(h1_pred, h_true)

        # --- C. Orthogonality Loss ---
        I_mat = torch.eye(N_r, device=device).unsqueeze(0).repeat(3, 1, 1)
        loss3_1 = torch.norm(z[0] - I_mat[0])
        loss3_2 = torch.norm(z[1] - I_mat[1])
        loss3_3 = torch.norm(z[2] - I_mat[2])
        loss3 = (loss3_1 + loss3_2 + loss3_3) / 3.0

        # --- D. Covariance Fitting Loss ---
        R_real_dev = R_real.to(device)
        loss4_1 = torch.norm(Learned_R[0] - R_real_dev[0])
        loss4_2 = torch.norm(Learned_R[1] - R_real_dev[1])
        loss4_3 = torch.norm(Learned_R[2] - R_real_dev[2])
        loss4 = (loss4_1 + loss4_2 + loss4_3) / 3.0

        # --- Combined Total Loss ---
        loss = loss1 + 10.0 * loss2 + 0.1 * loss3 + 0.1 * loss4

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_epoch_loss += loss.item()

    return total_epoch_loss / len(dataloader)


# ==========================================
# 5. Evaluation Loop (NMSE & SER)
# ==========================================
def evaluate_model(model, dataloader, N_r, device):
    model.eval()
    total_nmse = 0.0
    total_symbol_errors = 0
    total_symbols = 0

    with torch.no_grad():
        for zd, zp, xp, h_true, Xd_real, Xd_img in dataloader:
            batch_size = zd.size(0)

            zd, zp, xp = zd.to(device), zp.to(device), xp.to(device)
            h_true = h_true.to(device)
            Xd_real, Xd_img = Xd_real.to(device), Xd_img.to(device)

            h1_init = torch.zeros(batch_size, 3, 2 * N_r, 1, device=device)
            E_real_init = torch.zeros(batch_size, 3, 1, zd.size(3), device=device)
            E_img_init  = torch.zeros(batch_size, 3, 1, zd.size(3), device=device)

            F_concat, h1_pred, z, Learned_R, E_real_pred, E_img_pred = model(
                zd, zp, xp, h1_init, E_real_init, E_img_init, batch_size
            )

            # --- Channel Estimate NMSE Calculation ---
            mse = torch.sum((h1_pred - h_true) ** 2)
            power = torch.sum(h_true ** 2)
            total_nmse += (mse / power).item()

            # --- Symbol Error Rate (SER) Hard Decision ---
            pred_real = torch.sign(E_real_pred)
            pred_img  = torch.sign(E_img_pred)

            true_real = torch.sign(Xd_real)
            true_img  = torch.sign(Xd_img)

            real_errors = (pred_real != true_real)
            img_errors  = (pred_img != true_img)
            
            symbol_errors = (real_errors | img_errors).sum().item()
            
            total_symbol_errors += symbol_errors
            total_symbols += Xd_real.numel()

    avg_nmse = total_nmse / len(dataloader)
    ser = total_symbol_errors / total_symbols

    return avg_nmse, ser


# ==========================================
# 6. Hyperparameter Configuration
# ==========================================
NUM_LAYERS = 5
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3

# Channel & System Dimensions
N_R = 8         # Number of receive antennas
T_D = 10        # Data payload length
T_P = 4         # Pilot sequence length
SIGMA_2_W = 0.1 # Noise variance

# Initialize constant tensors required by Decoder submodules
all_ones = torch.ones(BATCH_SIZE, 3, 1, T_D, device=device)
P_s_init = torch.zeros(BATCH_SIZE, 3, 1, T_D, device=device)

# QPSK Reference Signal Constellations (Real + Imaginary blocks)
S0 = torch.randn(1, 2 * T_D, 1, device=device)
S1 = torch.randn(1, 2 * T_D, 1, device=device)
S2 = torch.randn(1, 2 * T_D, 1, device=device)
S3 = torch.randn(1, 2 * T_D, 1, device=device)

# Covariance matrix initialization
R_real = torch.eye(2 * N_R).repeat(3, 1, 1)


# ==========================================
# 7. Dataset Setup Helper
# ==========================================
def create_dataloaders(num_samples=1000, batch_size=BATCH_SIZE):
    # Generating dummy dataset matching model dimension specifications
    zd_data = torch.randn(num_samples, 3, 2 * N_R, T_D)
    zp_data = torch.randn(num_samples, 3, 2 * N_R, T_P)
    xp_data = torch.randn(num_samples, 3, 2, T_P)
    h_true_data = torch.randn(num_samples, 3, 2 * N_R, 1)
    Xd_real_data = torch.randn(num_samples, 3, 1, T_D)
    Xd_img_data = torch.randn(num_samples, 3, 1, T_D)

    dataset = TensorDataset(zd_data, zp_data, xp_data, h_true_data, Xd_real_data, Xd_img_data)
    
    train_size = int(0.8 * num_samples)
    test_size = num_samples - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ==========================================
# 8. Main Script Execution
# ==========================================
if __name__ == '__main__':
    print(f"Initializing Deep Unfolding Model on device: {device}")

    # 1. Instantiate Data Loaders
    train_loader, test_loader = create_dataloaders(num_samples=1280, batch_size=BATCH_SIZE)

    # 2. Instantiate Model, Loss Functions, and Optimizer
    model = DeepUnfoldingModel(
        num_layers=NUM_LAYERS,
        all_ones=all_ones,
        P_s_init=P_s_init,
        S0=S0, S1=S1, S2=S2, S3=S3,
        sigma_2_w=SIGMA_2_W,
        N_r=N_R,
        t_d=T_D,
        t_p=T_P
    ).to(device)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Main Training & Evaluation Loop
    print("\n--- Starting Training Loop ---")
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion_ce=criterion_ce,
            criterion_mse=criterion_mse,
            R_real=R_real,
            N_r=N_R,
            device=device
        )

        if epoch % 5 == 0 or epoch == 1:
            nmse, ser = evaluate_model(model, test_loader, N_R, device)
            print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Loss: {loss:.4f} | Test NMSE: {nmse:.4e} | Test SER: {ser:.4f}")

    print("\n--- Training Complete ---")