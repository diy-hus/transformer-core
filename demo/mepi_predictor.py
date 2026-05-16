#!/usr/bin/env python3
"""
MEPI frequency screening predictor v3

Fixes vs v1/v2:
  - Adds transformer geometry fields: outer diameter, inner diameter, height.
  - Computes outer radius, inner radius, radial thickness, and effective cross-sectional area.
  - Adds prediction mode:
      * closest_frequency: show prediction for the CSV row closest to target frequency
      * best_score: show the best row according to score = high Eff + high RUL + low Loss
      * override_frequency: replace the first tabular row's measured frequency by target frequency
  - Exports all row-level predictions.
  - Adds explicit GUI/CLI note that geometry parameters are used only for display/export.

Important model limitation:
  The supplied best.pth contains encoder/backbone/physics/core_head/rul_head only.
  It does NOT contain an efficiency prediction head. Therefore:
      Loss and RUL proxy are inferred from the model.
      Efficiency is calculated from the tabular CSV as: Eff = Vout*Iout/(Vin*Iin).
  Geometry parameters are used for display/export only.
  Current model input does not include geometry features.
  The supplied checkpoint expects 12 tabular inputs because encoder.tab_mlp.0.weight has input dimension 12.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception:
    tk = None

# -------------------------
# Configuration
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVEFORM_LEN = 4096
TAB_DIM = 12
HIDDEN_DIM = 256
GRU_HIDDEN = 128
GRU_LAYERS = 4
EPS = 1e-8

FEATURE_NAMES = [
    "VCF IN (V)",
    "Measured frequency (Hz)",
    "Phase (°)",
    "Vin (Vrms)",
    "Vout (Vrms)",
    "Iin (Arms)",
    "Iout (Arms)",
    "Room Temperature (°C)",
    "Transformer Temperature (°C)",
    "Vin Peak-Peak (V)",
    "Vout Peak-Peak (V)",
    "Load_R_ohm",
]

# -------------------------
# Data classes
# -------------------------
@dataclass
class Geometry:
    outer_diameter_mm: float = 34.58
    inner_diameter_mm: float = 20.81
    height_mm: float = 17.68

    @property
    def outer_radius_mm(self) -> float:
        return self.outer_diameter_mm / 2.0

    @property
    def inner_radius_mm(self) -> float:
        return self.inner_diameter_mm / 2.0

    @property
    def radial_thickness_mm(self) -> float:
        return max((self.outer_diameter_mm - self.inner_diameter_mm) / 2.0, 0.0)

    @property
    def effective_area_mm2(self) -> float:
        # Toroidal core cross-section approximation: radial thickness × height.
        return self.radial_thickness_mm * self.height_mm

    @property
    def effective_area_m2(self) -> float:
        return self.effective_area_mm2 * 1e-6


# -------------------------
# Helper functions
# -------------------------
def extract_float(x, default=np.nan) -> float:
    """Extract the last float from strings like '#800233323  1.05308e+01'."""
    if x is None:
        return default
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not matches:
        return default
    try:
        return float(matches[-1])
    except Exception:
        return default


def read_csv_rows(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
        if not rows:
            raise ValueError(f"CSV file has no data rows: {path}")
        return reader.fieldnames or [], rows


def resample_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.zeros(target_len, dtype=np.float32)
    old = np.linspace(0, 1, len(x), dtype=np.float32)
    new = np.linspace(0, 1, target_len, dtype=np.float32)
    return np.interp(new, old, x).astype(np.float32)


def normalize_waveform(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.nanmean(x)
    std = np.nanstd(x)
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return (x / std).astype(np.float32)


def load_waveform_csv(path: str, channel: str = "vin", target_len: int = WAVEFORM_LEN) -> np.ndarray:
    header, rows = read_csv_rows(path)
    lower_map = {h.lower().strip(): h for h in header}

    def find_col(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        simplified_candidates = [c.lower().replace("(", "").replace(")", "").replace(" ", "") for c in candidates]
        for h in header:
            hs = h.lower().replace("(", "").replace(")", "").replace(" ", "")
            if any(c in hs for c in simplified_candidates):
                return h
        return None

    vin_col = find_col(["Vin(V)", "Vin", "V_in", "input_voltage", "input"])
    vout_col = find_col(["Vout(V)", "Vout", "V_out", "output_voltage", "output"])

    channel = channel.lower().strip()
    if channel == "diff":
        if vin_col is None or vout_col is None:
            raise ValueError("channel='diff' requires both Vin and Vout columns in waveform CSV.")
        vin = np.array([extract_float(r.get(vin_col)) for r in rows], dtype=np.float32)
        vout = np.array([extract_float(r.get(vout_col)) for r in rows], dtype=np.float32)
        x = vin - vout
    elif channel == "vout":
        if vout_col is None:
            raise ValueError(f"Cannot find Vout column in waveform CSV. Columns: {header}")
        x = np.array([extract_float(r.get(vout_col)) for r in rows], dtype=np.float32)
    else:
        if vin_col is None:
            # fallback: second numeric column
            numeric_cols = []
            for h in header:
                vals = np.array([extract_float(r.get(h)) for r in rows[:30]], dtype=float)
                if np.isfinite(vals).sum() > 5:
                    numeric_cols.append(h)
            if len(numeric_cols) < 2:
                raise ValueError(f"Cannot find Vin column in waveform CSV. Columns: {header}")
            vin_col = numeric_cols[1]
        x = np.array([extract_float(r.get(vin_col)) for r in rows], dtype=np.float32)

    return normalize_waveform(resample_1d(x, target_len)).reshape(1, target_len)


def row_to_features(r: Dict[str, str], override_frequency: Optional[float] = None) -> List[float]:
    vcf = extract_float(r.get("VCF IN (V)"), 0.0)
    freq = extract_float(r.get("Measured frequency (Hz)"), 0.0)
    if override_frequency is not None and np.isfinite(override_frequency):
        freq = float(override_frequency)
    phase = extract_float(r.get("Phase (°)"), 0.0)
    vin = extract_float(r.get("Vin (Vrms)"), 0.0)
    vout = extract_float(r.get("Vout (Vrms)"), 0.0)
    iin = extract_float(r.get("Iin (Arms)"), 0.0)
    iout = extract_float(r.get("Iout (Arms)"), 0.0)
    room_t = extract_float(r.get("Room Temperature (°C)"), 0.0)
    trans_t = extract_float(r.get("Transformer Temperature (°C)"), 0.0)
    vin_pp = extract_float(r.get("Vin Peak-Peak (V)"), 0.0)
    vout_pp = extract_float(r.get("Vout Peak-Peak (V)"), 0.0)
    load_r = vout / (iout + EPS)
    return [vcf, freq, phase, vin, vout, iin, iout, room_t, trans_t, vin_pp, vout_pp, load_r]


def build_tabular_features(tabular_csv: str, mode: str = "closest_frequency", target_frequency: Optional[float] = None) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    _, rows = read_csv_rows(tabular_csv)
    mode = mode.lower().strip()

    if mode == "override_frequency":
        if target_frequency is None or not np.isfinite(target_frequency):
            raise ValueError("override_frequency mode requires a target frequency.")
        base_row = rows[0].copy()
        base_row["Measured frequency (Hz)"] = str(float(target_frequency))
        rows_for_prediction = [base_row]
        feats = [row_to_features(base_row, override_frequency=float(target_frequency))]
    else:
        rows_for_prediction = rows
        feats = [row_to_features(r) for r in rows_for_prediction]

    arr = np.asarray(feats, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != TAB_DIM:
        raise ValueError(f"Tabular feature shape must be [N,{TAB_DIM}], got {arr.shape}")
    return arr, rows_for_prediction


def safe_efficiency_from_row(r: Dict[str, str]) -> float:
    vin = extract_float(r.get("Vin (Vrms)"), np.nan)
    vout = extract_float(r.get("Vout (Vrms)"), np.nan)
    iin = extract_float(r.get("Iin (Arms)"), np.nan)
    iout = extract_float(r.get("Iout (Arms)"), np.nan)
    pin = vin * iin
    pout = vout * iout
    if not np.isfinite(pin) or abs(pin) < EPS:
        return np.nan
    return float(pout / pin)


def minmax_score(eff, loss, rul):
    def high(x):
        x = np.asarray(x, dtype=np.float32)
        return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + EPS)
    def low(x):
        return 1.0 - high(x)
    return high(eff) + high(rul) + low(loss)


def select_result(results: List[Dict], mode: str, target_frequency: Optional[float]) -> Dict:
    if not results:
        raise ValueError("No prediction results.")
    mode = mode.lower().strip()
    if mode == "best_score":
        return max(results, key=lambda x: x["screening_score"])
    if mode in ["closest_frequency", "override_frequency"] and target_frequency is not None and np.isfinite(target_frequency):
        return min(results, key=lambda x: abs(x["frequency_hz"] - target_frequency))
    return results[0]

# -------------------------
# Model reconstruction
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, max_len=10, d_model=256):
        super().__init__()
        self.register_buffer("pe", torch.zeros(max_len, d_model), persistent=True)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class MEPIEncoder(nn.Module):
    def __init__(self, tab_dim=12, hidden_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.wave_proj = nn.Linear(129, hidden_dim)
        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.pos_encoder = PositionalEncoding(max_len=10, d_model=hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

    def forward(self, waveform, tabular):
        cnn_feat = self.cnn(waveform).squeeze(-1)
        freq_scaled = tabular[:, 1:2] / 1000.0
        wave_feat = torch.cat([cnn_feat, freq_scaled], dim=1)
        wave_tok = self.wave_proj(wave_feat).unsqueeze(1)
        tab_tok = self.tab_mlp(tabular).unsqueeze(1)
        tab_tok = self.pos_encoder(tab_tok)
        fused, _ = self.cross_attn(wave_tok, tab_tok, tab_tok, need_weights=False)
        return fused


class BiGRUBackbone(nn.Module):
    def __init__(self, input_dim=256, hidden_size=128, num_layers=4):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out.mean(dim=1)


class PhysicsLayer(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.bottleneck = nn.Linear(hidden_dim, 5)
        self.residual_proj = nn.Linear(2, hidden_dim)

    def forward(self, s, tabular):
        aux = self.bottleneck(s)
        residual_2 = torch.tanh(aux[:, :2])
        correction = torch.tanh(self.residual_proj(residual_2))
        return s + correction


class MEPIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MEPIEncoder(TAB_DIM, HIDDEN_DIM)
        self.backbone = BiGRUBackbone(HIDDEN_DIM, GRU_HIDDEN, GRU_LAYERS)
        self.physics = PhysicsLayer(HIDDEN_DIM)
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.core_head = nn.Sequential(nn.Linear(HIDDEN_DIM, 64), nn.GELU(), nn.Linear(64, 1))
        self.rul_head = nn.Sequential(nn.Linear(HIDDEN_DIM, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, waveform, tabular):
        z = self.encoder(waveform, tabular)
        s = self.backbone(z)
        s = self.physics(s, tabular)
        s = self.norm(s)
        core_loss = F.softplus(self.core_head(s)).squeeze(-1)
        rul_out = self.rul_head(s)
        rul_mu = rul_out[:, 0]
        rul_var = F.softplus(rul_out[:, 1]) + 1e-6
        return {"loss": core_loss, "rul": rul_mu, "rul_var": rul_var}


def load_mepi_model(model_path: str, device: str = DEVICE) -> MEPIModel:
    obj = torch.load(model_path, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    elif isinstance(obj, dict) and "model_state_dict" in obj:
        sd = obj["model_state_dict"]
    else:
        sd = obj
    if not isinstance(sd, dict):
        raise RuntimeError("best.pth is not a state_dict. Please export a state_dict or TorchScript model.")
    clean = {k.replace("module.", ""): v for k, v in sd.items()}
    model = MEPIModel().to(device)
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing:
        print("Warning: missing keys:", missing)
    if unexpected:
        print("Warning: unexpected keys:", unexpected)
    model.eval()
    return model


def predict_all(
    model_path: str,
    tabular_csv: str,
    waveform_csv: str,
    waveform_channel: str = "vin",
    mode: str = "closest_frequency",
    target_frequency: Optional[float] = None,
    geometry: Optional[Geometry] = None,
    out_csv: Optional[str] = None,
) -> List[Dict]:
    geometry = geometry or Geometry()
    model = load_mepi_model(model_path, DEVICE)
    tab_np, rows = build_tabular_features(tabular_csv, mode=mode, target_frequency=target_frequency)
    wave_np = load_waveform_csv(waveform_csv, channel=waveform_channel, target_len=WAVEFORM_LEN)
    wave_batch = np.repeat(wave_np[None, :, :], repeats=tab_np.shape[0], axis=0).astype(np.float32)

    tab_t = torch.tensor(tab_np, dtype=torch.float32, device=DEVICE)
    wave_t = torch.tensor(wave_batch, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        out = model(wave_t, tab_t)
    pred_loss = out["loss"].detach().cpu().numpy().astype(float)
    pred_rul = out["rul"].detach().cpu().numpy().astype(float)
    pred_rul_var = out["rul_var"].detach().cpu().numpy().astype(float)

    eff = np.array([safe_efficiency_from_row(r) for r in rows], dtype=float)
    if mode == "override_frequency":
        # Efficiency remains calculated from the first row's measured RMS values.
        eff = np.asarray(eff, dtype=float)
    score = minmax_score(eff, pred_loss, pred_rul) if len(rows) > 1 else np.ones(len(rows), dtype=float)

    results = []
    for i, r in enumerate(rows):
        freq = extract_float(r.get("Measured frequency (Hz)"), np.nan)
        if mode == "override_frequency" and target_frequency is not None:
            freq = float(target_frequency)
        results.append({
            "row": i + 1,
            "time": r.get("Time", ""),
            "frequency_hz": freq,
            "outer_diameter_mm": geometry.outer_diameter_mm,
            "inner_diameter_mm": geometry.inner_diameter_mm,
            "height_mm": geometry.height_mm,
            "outer_radius_mm": geometry.outer_radius_mm,
            "inner_radius_mm": geometry.inner_radius_mm,
            "effective_area_mm2": geometry.effective_area_mm2,
            "efficiency_from_measurement": eff[i],
            "predicted_loss": pred_loss[i],
            "predicted_rul_proxy": pred_rul[i],
            "predicted_rul_variance": pred_rul_var[i],
            "screening_score": score[i],
            "geometry_note": "Geometry parameters are used for display/export only. Current model input does not include geometry features.",
        })

    if out_csv and results:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    return results

# -------------------------
# GUI
# -------------------------
class PredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MEPI Frequency Screening Predictor v3")
        self.root.geometry("1120x720")
        self.root.configure(bg="white")

        self.model_path = tk.StringVar(value="")
        self.tabular_path = tk.StringVar(value="")
        self.waveform_path = tk.StringVar(value="")
        self.channel = tk.StringVar(value="vin")
        self.mode = tk.StringVar(value="closest_frequency")
        self.target_freq = tk.StringVar(value="4000")

        self.outer_d = tk.StringVar(value="34.58")
        self.inner_d = tk.StringVar(value="20.81")
        self.height = tk.StringVar(value="17.68")
        self.geometry_info = tk.StringVar(value="-")

        self.loss_var = tk.StringVar(value="-")
        self.eff_var = tk.StringVar(value="-")
        self.rul_var = tk.StringVar(value="-")
        self.freq_var = tk.StringVar(value="-")
        self.mode_var = tk.StringVar(value="-")
        self.status = tk.StringVar(value=f"Ready. Device: {DEVICE}")

        self._build()
        self.update_geometry_info()

    def _build(self):
        tk.Label(self.root, text="Transformer Screening Frequency", font=("Arial", 22, "bold"), bg="white").pack(pady=16)
        main = tk.Frame(self.root, bg="white")
        main.pack(fill="both", expand=True, padx=32, pady=8)

        top = tk.Frame(main, bg="white")
        top.pack(fill="x", pady=8)

        input_box = tk.LabelFrame(top, text="INPUT", font=("Arial", 14, "bold"), bg="white", padx=14, pady=10, labelanchor="n")
        input_box.pack(side="left", fill="both", expand=True, padx=(0, 20))
        pred_box = tk.LabelFrame(top, text="PREDICTION", font=("Arial", 14, "bold"), bg="white", padx=14, pady=10, labelanchor="n")
        pred_box.pack(side="right", fill="both", expand=True, padx=(20, 0))

        self._file_row(input_box, "Model .pth", self.model_path, self.pick_model, 0)
        self._file_row(input_box, "Tabular CSV", self.tabular_path, self.pick_tabular, 1)
        self._file_row(input_box, "Waveform CSV", self.waveform_path, self.pick_waveform, 2)

        tk.Label(input_box, text="Waveform channel", font=("Arial", 11, "bold"), bg="white").grid(row=3, column=0, sticky="w", pady=6)
        ttk.Combobox(input_box, textvariable=self.channel, values=["vin", "vout", "diff"], width=18, state="readonly").grid(row=3, column=1, sticky="w", pady=6)

        tk.Label(input_box, text="Prediction mode", font=("Arial", 11, "bold"), bg="white").grid(row=4, column=0, sticky="w", pady=6)
        ttk.Combobox(
            input_box,
            textvariable=self.mode,
            values=["closest_frequency", "best_score", "override_frequency"],
            width=22,
            state="readonly",
        ).grid(row=4, column=1, sticky="w", pady=6)

        tk.Label(input_box, text="Target frequency (Hz)", font=("Arial", 11, "bold"), bg="white").grid(row=5, column=0, sticky="w", pady=6)
        tk.Entry(input_box, textvariable=self.target_freq, font=("Arial", 10, "bold"), width=20).grid(row=5, column=1, sticky="w", pady=6)

        # Geometry box
        geom = tk.LabelFrame(main, text="GEOMETRY", font=("Arial", 14, "bold"), bg="white", padx=14, pady=8, labelanchor="n")
        geom.pack(fill="x", pady=12)
        self._entry_row(geom, "Outer diameter OD (mm)", self.outer_d, 0, self.update_geometry_info)
        self._entry_row(geom, "Inner diameter ID (mm)", self.inner_d, 1, self.update_geometry_info)
        self._entry_row(geom, "Height H (mm)", self.height, 2, self.update_geometry_info)
        tk.Label(geom, text="Computed", font=("Arial", 11), bg="white").grid(row=3, column=0, sticky="w", pady=6)
        tk.Label(geom, textvariable=self.geometry_info, font=("Arial", 10, "bold"), bg="white", justify="left", anchor="w").grid(row=3, column=1, columnspan=3, sticky="w", pady=6)
        self.geometry_note = tk.StringVar(
            value=(
                "Geometry parameters are used for display/export only.\n"
                "Current model input does not include geometry features."
            )
        )
        tk.Label(
            geom,
            textvariable=self.geometry_note,
            font=("Arial", 10, "bold"),
            fg="#b05a00",
            bg="white",
            justify="left",
            anchor="w",
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=6)

        self._output_row(pred_box, "Selected frequency (Hz)", self.freq_var, 0)
        self._output_row(pred_box, "Eff from RMS", self.eff_var, 1)
        self._output_row(pred_box, "Predicted loss", self.loss_var, 2)
        self._output_row(pred_box, "Predicted RUL", self.rul_var, 3)
        self._output_row(pred_box, "Mode", self.mode_var, 4)

        btn_frame = tk.Frame(main, bg="white")
        btn_frame.pack(fill="x", pady=14)
        tk.Button(btn_frame, text="PREDICT", command=self.run_predict, font=("Arial", 14, "bold"), bg="#2e9ca6", fg="white", width=16, height=2).pack(side="left")
        tk.Button(btn_frame, text="Export CSV", command=self.export_csv, font=("Arial", 12, "bold"), width=12, height=2).pack(side="left", padx=12)

        tk.Label(self.root, textvariable=self.status, font=("Arial", 10, "bold"), bg="white", fg="#444", anchor="w").pack(fill="x", padx=32, pady=(0, 12))

    def _file_row(self, parent, label, var, command, row):
        tk.Label(parent, text=label, font=("Arial", 11, "bold"), bg="white").grid(row=row, column=0, sticky="w", pady=6)
        tk.Entry(parent, textvariable=var, font=("Arial", 10, "bold"), width=58).grid(row=row, column=1, sticky="ew", padx=8, pady=6)
        tk.Button(parent, text="Browse", command=command, width=8).grid(row=row, column=2, pady=6)
        parent.grid_columnconfigure(1, weight=1)

    def _entry_row(self, parent, label, var, row, callback=None):
        tk.Label(parent, text=label, font=("Arial", 11, "bold"), bg="white").grid(row=row, column=0, sticky="w", pady=5)
        e = tk.Entry(parent, textvariable=var, font=("Arial", 10, "bold"), width=16)
        e.grid(row=row, column=1, sticky="w", pady=5)
        if callback:
            e.bind("<KeyRelease>", lambda _event: callback())

    def _output_row(self, parent, label, var, row):
        tk.Label(parent, text=label, font=("Arial", 12), bg="white").grid(row=row, column=0, sticky="w", pady=8)
        tk.Label(parent, textvariable=var, font=("Arial", 13, "bold"), bg="white").grid(row=row, column=1, sticky="e", padx=12, pady=8)
        parent.grid_columnconfigure(1, weight=1)

    def pick_model(self):
        p = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pth *.pt"), ("All files", "*.*")])
        if p:
            self.model_path.set(p)

    def pick_tabular(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if p:
            self.tabular_path.set(p)

    def pick_waveform(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if p:
            self.waveform_path.set(p)

    def parse_geometry(self) -> Geometry:
        return Geometry(
            outer_diameter_mm=extract_float(self.outer_d.get(), 34.58),
            inner_diameter_mm=extract_float(self.inner_d.get(), 20.81),
            height_mm=extract_float(self.height.get(), 17.68),
        )

    def update_geometry_info(self):
        g = self.parse_geometry()
        self.geometry_info.set(
            f"Outer radius = {g.outer_radius_mm:.3f} mm; Inner radius = {g.inner_radius_mm:.3f} mm; "
            f"Radial thickness = {g.radial_thickness_mm:.3f} mm; Ae ≈ {g.effective_area_mm2:.3f} mm²"
        )

    def target_frequency_value(self) -> Optional[float]:
        f = extract_float(self.target_freq.get(), np.nan)
        return float(f) if np.isfinite(f) else None

    def _validate_paths(self):
        for name, p in [("model", self.model_path.get()), ("tabular", self.tabular_path.get()), ("waveform", self.waveform_path.get())]:
            if not p or not Path(p).exists():
                raise FileNotFoundError(f"Please select a valid {name} file.")

    def run_predict(self):
        try:
            self._validate_paths()
            mode = self.mode.get()
            target_f = self.target_frequency_value()
            results = predict_all(
                self.model_path.get(),
                self.tabular_path.get(),
                self.waveform_path.get(),
                waveform_channel=self.channel.get(),
                mode=mode,
                target_frequency=target_f,
                geometry=self.parse_geometry(),
            )
            selected = select_result(results, mode=mode, target_frequency=target_f)
            self.freq_var.set(f'{selected["frequency_hz"]:.3f}')
            self.eff_var.set(f'{selected["efficiency_from_measurement"]:.6f}')
            self.loss_var.set(f'{selected["predicted_loss"]:.6f}')
            self.rul_var.set(f'{selected["predicted_rul_proxy"]:.6f}')
            self.mode_var.set(mode)
            self.status.set(
                f'Prediction completed. Selected row {selected["row"]}, frequency {selected["frequency_hz"]:.3f} Hz. '
                'Geometry is display/export only.'
            )
        except Exception as e:
            messagebox.showerror("Prediction error", str(e))
            self.status.set("Prediction failed.")

    def export_csv(self):
        try:
            self._validate_paths()
            out = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile="mepi_predictions_v2.csv")
            if not out:
                return
            predict_all(
                self.model_path.get(),
                self.tabular_path.get(),
                self.waveform_path.get(),
                waveform_channel=self.channel.get(),
                mode=self.mode.get(),
                target_frequency=self.target_frequency_value(),
                geometry=self.parse_geometry(),
                out_csv=out,
            )
            self.status.set(f"Saved predictions to {out}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))
            self.status.set("Export failed.")

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Run command-line prediction instead of GUI")
    parser.add_argument("--model", default="best.pth")
    parser.add_argument("--tabular", default="Data_20260427_16h21p.csv")
    parser.add_argument("--waveform", default="Waveform_3026_500Hz_16h14h57m.csv")
    parser.add_argument("--channel", default="vin", choices=["vin", "vout", "diff"])
    parser.add_argument("--mode", default="closest_frequency", choices=["closest_frequency", "best_score", "override_frequency"])
    parser.add_argument("--frequency", type=float, default=4000.0)
    parser.add_argument("--outer-diameter", type=float, default=34.58)
    parser.add_argument("--inner-diameter", type=float, default=20.81)
    parser.add_argument("--height", type=float, default=17.68)
    parser.add_argument("--out", default="mepi_predictions_v2.csv")
    args = parser.parse_args()

    geom = Geometry(args.outer_diameter, args.inner_diameter, args.height)

    if args.cli:
        results = predict_all(
            args.model,
            args.tabular,
            args.waveform,
            waveform_channel=args.channel,
            mode=args.mode,
            target_frequency=args.frequency,
            geometry=geom,
            out_csv=args.out,
        )
        selected = select_result(results, mode=args.mode, target_frequency=args.frequency)
        print("\nSelected candidate")
        print("------------------")
        print(f"Mode:        {args.mode}")
        print(f"Row:         {selected['row']}")
        print(f"Frequency:   {selected['frequency_hz']:.3f} Hz")
        print(f"Eff RMS:     {selected['efficiency_from_measurement']:.6f}")
        print(f"Loss pred:   {selected['predicted_loss']:.6f}")
        print(f"RUL pred:    {selected['predicted_rul_proxy']:.6f}")
        print(f"RUL var:     {selected['predicted_rul_variance']:.6f}")
        print(f"OD/ID/H:     {geom.outer_diameter_mm:.3f}/{geom.inner_diameter_mm:.3f}/{geom.height_mm:.3f} mm")
        print(f"Radii:       {geom.outer_radius_mm:.3f}/{geom.inner_radius_mm:.3f} mm")
        print(f"Ae approx:   {geom.effective_area_mm2:.3f} mm^2")
        print("Note: Geometry parameters are used for display/export only.")
        print("      Current model input does not include geometry features.")
        print(f"\nSaved all predictions to: {args.out}")
        return

    if tk is None:
        print("Tkinter is not available. Use --cli mode.")
        sys.exit(1)
    root = tk.Tk()
    PredictorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
