import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import logging
import csv
import numpy as np
import traceback

class MagNetH5Dataset(Dataset):
    def __init__(self, h5_path, normalize=True):
        self.h5_path = h5_path
        self.normalize = normalize
        
        with h5py.File(h5_path, 'r') as f:
            self.length = len(f['P'])
            self.materials = np.unique(f['material'][:]).tolist()
            self.mat_to_idx = {m.decode('utf-8'): i for i, m in enumerate(self.materials)}
            
            if normalize:
                self.B_mean = float(f['B'][:].mean())
                self.B_std  = float(f['B'][:].std()) + 1e-8
                self.f_mean = float(f['f'][:].mean())
                self.f_std  = float(f['f'][:].std()) + 1e-8
                self.T_mean = float(f['T'][:].mean())
                self.T_std  = float(f['T'][:].std()) + 1e-8
                self.P_log_mean = float(np.log1p(f['P'][:]).mean())
                self.P_log_std  = float(np.log1p(f['P'][:]).std()) + 1e-8

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            B = torch.from_numpy(f['B'][idx].astype(np.float32)).unsqueeze(0)
            f_val = torch.tensor([f['f'][idx]], dtype=torch.float32)
            T_val = torch.tensor([f['T'][idx]], dtype=torch.float32)
            P_raw = torch.tensor([f['P'][idx]], dtype=torch.float32)
            
            mat_name = f['material'][idx].decode('utf-8')
            mat_onehot = F.one_hot(torch.tensor(self.mat_to_idx[mat_name]), num_classes=10).float()
            tabular = torch.cat([f_val, T_val, mat_onehot])

            if self.normalize:
                B = (B - self.B_mean) / self.B_std
                tabular[0] = (tabular[0] - self.f_mean) / self.f_std
                tabular[1] = (tabular[1] - self.T_mean) / self.T_std
                P = torch.log1p(P_raw)
                P = (P - self.P_log_mean) / self.P_log_std
            else:
                P = P_raw

        return B, tabular, P

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)

class MultimodalInputEncoder(nn.Module):
    def __init__(self, tabular_dim=12, embed_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 7, padding=3), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 7, padding=3), nn.GELU()
        )
        self.wave_proj = nn.Linear(128 + 1, embed_dim)
        
        self.tab_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 256), 
            nn.GELU(), 
            nn.BatchNorm1d(256), 
            nn.Linear(256, embed_dim)
        )
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=10)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

    def forward(self, waveform, tabular):
        cnn_feat = self.cnn(waveform)
        seq_len_out = cnn_feat.shape[-1]
        
        fft_complex = torch.fft.rfft(waveform, dim=-1)
        fft_mag = torch.abs(fft_complex)
        fft_mag_pooled = F.adaptive_avg_pool1d(fft_mag, seq_len_out)
        
        wave_concat = torch.cat([cnn_feat, fft_mag_pooled], dim=1).transpose(1, 2)
        query = self.wave_proj(wave_concat)
        
        tab_feat = self.tab_mlp(tabular).unsqueeze(1)
        key_value = self.pos_encoder(tab_feat)
        
        fused, _ = self.cross_attn(query=query, key=key_value, value=key_value)
        return fused

class PhysicsInformedResidualLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.bottleneck = nn.Linear(d_model, 5)
        self.residual_proj = nn.Linear(2, d_model)

    def forward(self, x, waveform, tabular):
        phys = F.softplus(self.bottleneck(x))
        k, alpha, beta, A, Ea = phys.unbind(-1)
        B_max = torch.max(torch.abs(waveform), dim=-1)[0].squeeze(1) + 1e-4
        f = F.softplus(tabular[:, 0]) + 1e-4
        T = F.softplus(tabular[:, 1]) + 1.0
        
        A = torch.clamp(A, min=1e-3, max=50.0)
        Ea = torch.clamp(Ea, min=0.1, max=15.0)
        arrhenius = torch.clamp(A * torch.exp(-Ea / T), max=500.0)
        steinmetz = torch.clamp(k * (f ** alpha) * (B_max ** beta), max=500.0)
        
        residuals = torch.stack([steinmetz, arrhenius], dim=-1)
        residuals = torch.clamp(residuals, -200.0, 200.0)
        return x + self.residual_proj(residuals)

class GRUBackbone(nn.Module):
    def __init__(self, d_model=256, num_layers=2, bidirectional=True):
        super().__init__()
        hidden_size = d_model // 2 if bidirectional else d_model
        self.gru = nn.GRU(d_model, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
    
    def forward(self, x):
        x, _ = self.gru(x)
        return x

class MagNetCoreLossModel(nn.Module):
    def __init__(self, d_model=256, num_layers=2):
        super().__init__()
        self.encoder = MultimodalInputEncoder(tabular_dim=12, embed_dim=d_model)
        self.backbone = GRUBackbone(d_model, num_layers=num_layers, bidirectional=True)
        self.physics = PhysicsInformedResidualLayer(d_model)
        self.norm = nn.LayerNorm(d_model)
        
        self.core_head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))
        self.rul_head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 2))
    
    def forward(self, waveform, tabular):
        x = self.encoder(waveform, tabular)
        x = self.backbone(x)
        x_phys = self.physics(x[:, -1, :], waveform, tabular) 
        x_out = self.norm(x_phys)
        
        core_pred = self.core_head(x_out).squeeze(-1)
        rul_pred = self.rul_head(x_out)
        return core_pred, rul_pred

def calculate_metrics(pred, target):
    pred = torch.clamp(torch.nan_to_num(pred, 0.0), -10, 10)
    target = torch.nan_to_num(target, 0.0)
    mae = F.l1_loss(pred, target).item()
    rmse = torch.sqrt(F.mse_loss(pred, target)).item()
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = float(torch.clamp(1 - ss_res / ss_tot, -100, 1)) if ss_tot > 1e-8 else 0.0
    rel_err = torch.abs((target - pred) / (torch.abs(target) + 1e-8)) * 100
    p95 = torch.quantile(rel_err, 0.95).item()
    return mae, rmse, r2, p95

def pretrain_bigru_layer_experiment(num_layers, epochs=10, batch_size=48, lr=3e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = f"BiGRU_{num_layers}L"
    checkpoint_dir = f"checkpoints_exp2_pretrain/{exp_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    log_file = f"{checkpoint_dir}/training_log.txt"
    logger = logging.getLogger(f"Pretrain_{exp_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    csv_path = f"{checkpoint_dir}/metrics.csv"
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    writer.writerow(["Step", "Loss", "MAE", "RMSE", "R2", "95th_Pct_Error", "GradNorm"])
    
    logger.info(f"STARTING PRETRAIN EXPERIMENT 2: {exp_name} | Device: {device} | Batch: {batch_size} | LR: {lr}")
    
    data_path = "/mnt/disk1/Dataset for Research/May_bien_ap/magnet_pretrain_186k.h5"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Pretrain data not found at {data_path}")
        
    dataset = MagNetH5Dataset(data_path, normalize=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True, drop_last=True)
    
    model = MagNetCoreLossModel(d_model=256, num_layers=num_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=len(loader)*epochs,
        pct_start=0.1, anneal_strategy='cos', div_factor=25.0
    )
    
    best_mae = float('inf')
    best_path = f"{checkpoint_dir}/best.pth"
    global_step = 0
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} [{exp_name}]")
        
        for B, tabular, P in pbar:
            B = B.to(device, non_blocking=True)
            tabular = tabular.to(device, non_blocking=True)
            P = P.to(device, non_blocking=True).view(-1)
            
            optimizer.zero_grad()
            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                core_pred, _ = model(B, tabular)
                loss = F.mse_loss(core_pred, P) + 0.3 * F.l1_loss(core_pred, P)
            
            scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            global_step += 1
            
            if global_step % 100 == 0:
                mae, rmse, r2, p95 = calculate_metrics(core_pred, P)
                logger.info(f"Step {global_step:5d} | Loss: {loss.item():.4f} | "
                           f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f} | "
                           f"95th: {p95:.2f}% | GradNorm: {grad_norm:.2f}")
                writer.writerow([global_step, loss.item(), mae, rmse, r2, p95, grad_norm])
                csv_file.flush()
                
                if mae < best_mae:
                    best_mae = mae
                    torch.save(model.state_dict(), best_path)
                    logger.info(f"BEST MODEL UPDATED! MAE = {best_mae:.4f} (saved to {best_path})")
    
    csv_file.close()
    logger.info(f"COMPLETED {exp_name} | Best MAE = {best_mae:.4f}")
    return best_mae

if __name__ == "__main__":
    layer_configs = [2, 4, 6, 8, 10]
    
    results = {}
    print("\n=== STARTING EXPERIMENT 2: PRE-TRAIN BiGRU LAYERS ABLATION ===")
    
    for layers in layer_configs:
        print(f"\n{'='*60}")
        print(f"RUNNING PRE-TRAIN: BiGRU with {layers} Layers")
        print(f"{'='*60}")
        
        try:
            mae = pretrain_bigru_layer_experiment(
                num_layers=layers, 
                epochs=10, 
                batch_size=48, 
                lr=3e-5
            )
            results[f"{layers}L"] = mae
        except Exception as e:
            print(f"Error during pre-train BiGRU {layers} layers: {e}")
            traceback.print_exc()
    
    print("\n" + "="*50)
    print("=== PRE-TRAINING COMPARISON RESULTS (BiGRU LAYERS) ===")
    print("="*50)
    for layers, mae in sorted(results.items(), key=lambda x: int(x[0].replace('L', ''))):
        print(f"BiGRU {layers:>3} -> Best MAE = {mae:.6f}")