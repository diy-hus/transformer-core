import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import logging
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class CobaltTransformerDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        
        # Waveform column
        waveform_candidates = ['B_waveform', 'b_waveform', 'waveform', 'B_wave']
        self.waveform_col = next((col for col in waveform_candidates if col in self.df.columns), None)
        
        self.B_waveforms = []
        for s in self.df[self.waveform_col].values:
            try:
                if isinstance(s, str):
                    arr = np.array(json.loads(s), dtype=np.float32)
                elif isinstance(s, (list, np.ndarray)):
                    arr = np.array(s, dtype=np.float32)
                else:
                    arr = np.zeros(1024, dtype=np.float32)
                self.B_waveforms.append(arr)
            except:
                self.B_waveforms.append(np.zeros(1024, dtype=np.float32))
        
        self.B_waveforms = np.array(self.B_waveforms, dtype=object)
        all_B = np.concatenate([x for x in self.B_waveforms if len(x) > 0])
        self.B_mean = all_B.mean()
        self.B_std = all_B.std() + 1e-8
        
        self.freq = self.df['frequency_hz'].values.astype(np.float32)
        self.temp = self.df['temperature_core_c'].values.astype(np.float32)
        
        if 'P_loss' in self.df.columns:
            p_loss = self.df['P_loss'].values
        else:
            eff = np.clip(self.df['efficiency_percent'].values / 100.0, 0.01, 0.999)
            p_loss = self.df['input_power_w'].values * (1 - eff)
        
        self.P_loss = np.log1p(p_loss).astype(np.float32)
        
        core_types = ['silicon_steel_laminated', 'commercial_nanocrystalline', 'soft_magnetic_cobalt_coated']
        self.core_indices = {ct: i for i, ct in enumerate(core_types)}
        self.core_onehot = np.zeros((len(self.df), len(core_types)), dtype=np.float32)
        for i, ct in enumerate(self.df['core_type']):
            idx = self.core_indices.get(ct, 2)
            self.core_onehot[i, idx] = 1.0

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        B_array = self.B_waveforms[idx]
        if len(B_array) < 1024:
            B_array = np.pad(B_array, (0, 1024 - len(B_array)), 'constant')
        else:
            B_array = B_array[:1024]
            
        B_array = (B_array - self.B_mean) / self.B_std
        B = torch.from_numpy(B_array.astype(np.float32)).unsqueeze(0)
        
        freq_norm = (self.freq[idx] - self.freq.mean()) / (self.freq.std() + 1e-8)
        temp_norm = (self.temp[idx] - self.temp.mean()) / (self.temp.std() + 1e-8)
        
        tabular_5 = np.concatenate([np.array([freq_norm, temp_norm]), self.core_onehot[idx]])
        tabular = np.concatenate([tabular_5, np.zeros(7, dtype=np.float32)])
        tabular = torch.from_numpy(tabular).float()
        
        P = torch.tensor(self.P_loss[idx]).float()
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
        return x + self.pe[:x.size(1), :].unsqueeze(0)


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
            nn.Linear(tabular_dim, 256), nn.GELU(), nn.BatchNorm1d(256), nn.Linear(256, embed_dim)
        )
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=10)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

    def forward(self, waveform, tabular):
        cnn_feat = self.cnn(waveform)
        fft_complex = torch.fft.rfft(waveform.float(), dim=-1)
        fft_mag_pooled = F.adaptive_avg_pool1d(torch.abs(fft_complex), cnn_feat.shape[-1])
        query = self.wave_proj(torch.cat([cnn_feat, fft_mag_pooled], dim=1).transpose(1, 2))
        key_value = self.pos_encoder(self.tab_mlp(tabular).unsqueeze(1))
        fused, _ = self.cross_attn(query=query, key=key_value, value=key_value)
        return fused


class PhysicsInformedResidualLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.bottleneck = nn.Linear(d_model, 5)
        self.residual_proj = nn.Linear(2, d_model)
        self.physics_residual = None  # [FIX 1] Biến lưu trữ residual

    def forward(self, x, waveform, tabular):
        k, alpha, beta, A, Ea = F.softplus(self.bottleneck(x)).unbind(-1)
        B_max = torch.max(torch.abs(waveform), dim=-1)[0].squeeze(1) + 1e-4
        f = F.softplus(tabular[:, 0]) + 1e-4
        T = F.softplus(tabular[:, 1]) + 1.0
        alpha = torch.clamp(alpha, max=3.0)
        beta = torch.clamp(beta, max=3.0)
        A = torch.clamp(A, min=1e-3, max=50.0)
        Ea = torch.clamp(Ea, min=0.1, max=15.0)
        arrhenius = torch.clamp(A * torch.exp(-Ea / T), max=500.0)
        steinmetz = torch.clamp(k * (f ** alpha) * (B_max ** beta), max=500.0)
        
        residuals = torch.clamp(torch.stack([steinmetz, arrhenius], dim=-1), -200.0, 200.0)
        self.physics_residual = residuals  # [FIX 1] Lưu lại để tính loss
        return x + self.residual_proj(residuals)


class GRUBackbone(nn.Module):
    def __init__(self, d_model=256, num_layers=6, bidirectional=True):
        super().__init__()
        hidden_size = d_model // 2 if bidirectional else d_model
        self.gru = nn.GRU(d_model, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        x, _ = self.gru(x)
        return x


class MagNetCoreLossModel(nn.Module):
    def __init__(self, d_model=256, num_layers=6):
        super().__init__()
        self.encoder = MultimodalInputEncoder(tabular_dim=12, embed_dim=d_model)
        self.backbone = GRUBackbone(d_model, num_layers=num_layers, bidirectional=True)
        self.physics = PhysicsInformedResidualLayer(d_model)
        self.norm = nn.LayerNorm(d_model)
        
        self.core_head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))
        self.unc_head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))  # [FIX 2] Head dự đoán độ bất định (log variance)
        self.rul_head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, waveform, tabular):
        x = self.encoder(waveform, tabular)
        x = self.backbone(x)
        x_phys = self.physics(x[:, -1, :], waveform, tabular)
        x_out = self.norm(x_phys)
        
        core_pred = self.core_head(x_out).squeeze(-1)
        unc_pred = self.unc_head(x_out).squeeze(-1)  # [FIX 2] Xuất giá trị uncertainty
        rul_pred = self.rul_head(x_out)
        return core_pred, unc_pred, rul_pred


def calculate_comprehensive_metrics(pred, target):
    pred = torch.clamp(torch.nan_to_num(pred, 0.0), -10, 10).view(-1)
    target = torch.nan_to_num(target, 0.0).view(-1)
    
    mae_norm = F.l1_loss(pred, target).item()
    rmse = torch.sqrt(F.mse_loss(pred, target)).item()
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = float(torch.clamp(1 - ss_res / ss_tot, -100, 1)) if ss_tot > 1e-8 else 0.0
    
    pred_watts = torch.expm1(pred)
    target_watts = torch.expm1(target)
    mae_watts = F.l1_loss(pred_watts, target_watts).item()
    rel_err = torch.abs((target_watts - pred_watts) / (torch.abs(target_watts) + 1e-8))
    p95 = torch.quantile(rel_err * 100, 0.95).item()
    mape = torch.mean(rel_err).item() * 100
    max_err = torch.max(torch.abs(target_watts - pred_watts)).item()
    
    return {
        "MAE_norm": mae_norm, "MAE_Watts": mae_watts, "RMSE": rmse,
        "R2": r2, "P95_Error_pct": p95, "MAPE": mape, "MaxErr": max_err
    }


def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    all_pred, all_target = [], []
    with torch.no_grad():
        for B, tabular, P in val_loader:
            B, tabular, P = B.to(device), tabular.to(device), P.to(device).view(-1)
            core_pred, _, _ = model(B, tabular)  # Unpack thêm unc_pred
            val_loss += (F.huber_loss(core_pred, P, delta=0.5) + 0.1 * F.l1_loss(core_pred, P)).item()
            all_pred.append(core_pred)
            all_target.append(P)
    metrics = calculate_comprehensive_metrics(torch.cat(all_pred), torch.cat(all_target))
    return val_loss / len(val_loader), metrics


def finetune_lambda_experiment(lambda3, lambda4, pretrained_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = f"BiGRU6L_l3_{lambda3}_l4_{lambda4}"
    save_dir = f"checkpoints_lambda_ablation/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)

    # Logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(f"{save_dir}/finetune_log.txt", encoding='utf-8'),
                                  logging.StreamHandler()])
    logger = logging.getLogger()

    # CSV
    csv_path = f"{save_dir}/finetune_metrics.csv"
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    writer.writerow(["Step", "Phase", "Loss", "MAE_norm", "MAE_Watts", "RMSE", "R2", 
                     "P95_Error_pct", "MAPE", "MaxErr", "lambda3", "lambda4"])

    # Data
    df = pd.read_csv("/mnt/disk1/Dataset for Research/May_bien_ap/finetune_data/finetune_dataset.csv")
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df.get('core_type'))
    
    train_loader = DataLoader(CobaltTransformerDataset(df_train), batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(CobaltTransformerDataset(df_val), batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    # Model
    model = MagNetCoreLossModel(d_model=256, num_layers=6).to(device)
    
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True), strict=False)
        logger.info(f"Successfully loaded pretrained weights from: {pretrained_path}")
    else:
        logger.error(f"Pretrained file not found at {pretrained_path}")
        return None

    optimizer = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-4)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    best_mae = float('inf')
    global_step = 0

    for epoch in range(50):
        model.train()
        for B, tabular, P in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            B, tabular, P = B.to(device), tabular.to(device), P.to(device).view(-1)
            global_step += 1

            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                core_pred, unc_pred, _ = model(B, tabular)
                
                data_loss = F.huber_loss(core_pred, P, delta=0.5) + 0.1 * F.l1_loss(core_pred, P)
                
                phys_residual = model.physics.physics_residual   
                phys_loss = torch.mean(torch.abs(phys_residual))
                
                mse_loss = F.mse_loss(core_pred, P, reduction='none')
                unc_loss = torch.mean(0.5 * torch.exp(-unc_pred) * mse_loss + 0.5 * unc_pred)
                
                total_loss = data_loss + lambda3 * phys_loss + lambda4 * unc_loss

            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad() # Bổ sung thêm zero_grad nếu bạn chưa clear grad ở đầu batch

            if global_step % 100 == 0:
                metrics = calculate_comprehensive_metrics(core_pred, P)
                writer.writerow([global_step, "TRAIN", total_loss.item(),
                                metrics["MAE_norm"], metrics["MAE_Watts"], metrics["RMSE"],
                                metrics["R2"], metrics["P95_Error_pct"], metrics["MAPE"],
                                metrics["MaxErr"], lambda3, lambda4])
                csv_file.flush()

        val_loss, v_metrics = validate_model(model, val_loader, device)
        writer.writerow([global_step, "VALID", val_loss, v_metrics["MAE_norm"], 
                        v_metrics["MAE_Watts"], v_metrics["RMSE"], v_metrics["R2"],
                        v_metrics["P95_Error_pct"], v_metrics["MAPE"], v_metrics["MaxErr"],
                        lambda3, lambda4])
        csv_file.flush()

        if v_metrics['MAE_norm'] < best_mae:
            best_mae = v_metrics['MAE_norm']
            torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
            logger.info(f"BEST MODEL UPDATED - MAE_norm: {best_mae:.4f} | λ₃={lambda3}, λ₄={lambda4}")

    csv_file.close()
    logger.info(f"Finished {exp_name} - Best MAE_norm: {best_mae:.4f}")
    
    logger.handlers.clear()
    
    return best_mae


# ========================== MAIN ==========================
if __name__ == "__main__":
    PRETRAINED_PATH = "/home/deltax/research/May_bien_ap/try_4/Experiment_2/checkpoints_exp2_pretrain/BiGRU_6L/best.pth"
    
    lambda_configs = [
        (0.1, 0.05), (0.1, 0.10), (0.1, 0.20),
        (0.3, 0.05), (0.3, 0.10), (0.3, 0.20),
        (0.5, 0.05), (0.5, 0.10), (0.5, 0.20),
        (1.0, 0.05), (1.0, 0.10), (1.0, 0.20)
    ]

    print("=== STARTING λ₃ AND λ₄ ABLATION STUDY (BiGRU 6 Layers) ===")
    
    for λ3, λ4 in lambda_configs:
        print(f"\nRunning experiment: λ₃ = {λ3}, λ₄ = {λ4}")
        best_mae = finetune_lambda_experiment(λ3, λ4, PRETRAINED_PATH)
        print(f"Completed → Best MAE_norm = {best_mae:.4f}\n")
    
    print("=== ABLATION STUDY ON λ₃ AND λ₄ COMPLETED ===")