# =====================================================
# IMPROVED QUANTUM AUTOENCODER (HYBRID QAE)
# 16 FEATURES -> 4 QUBITS -> NN DECODER
# =====================================================

import pennylane as qml
from pennylane import numpy as pnp
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import time

# =====================================================
# 1. CONFIG
# =====================================================
DATA_FILE = "qae_train_16fitur_scaled_10k.csv"

N_FEATURES = 16
N_QUBITS = 4

EPOCHS = 100
BATCH_SIZE = 32
LR = 0.005
PATIENCE = 10

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# 2. LOAD DATA
# =====================================================
df = pd.read_csv(DATA_FILE)
X = df.values.astype(np.float64)

# split train / val
split = int(0.8 * len(X))
X_train = torch.tensor(X[:split], dtype=torch.float64).to(DEVICE)
X_val   = torch.tensor(X[split:], dtype=torch.float64).to(DEVICE)

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# =====================================================
# 3. QUANTUM DEVICE
# =====================================================
dev = qml.device("default.qubit", wires=N_QUBITS)

# =====================================================
# 4. QUANTUM ENCODER (FIXED)
# =====================================================
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_encoder(x, weights):

    # 🔹 Angle encoding PER FEATURE
    for i in range(N_QUBITS):
        qml.RY(x[i] * pnp.pi, wires=i)
        qml.RZ(x[i + N_QUBITS] * pnp.pi, wires=i)

    # 🔹 Entanglement layer 1
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])

    # 🔹 Trainable layer
    idx = 0
    for i in range(N_QUBITS):
        qml.RY(weights[idx], wires=i)
        qml.RZ(weights[idx + 1], wires=i)
        idx += 2

    # 🔹 Entanglement layer 2
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# =====================================================
# 5. HYBRID QAE MODEL
# =====================================================
class HybridQAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.q_weights = nn.Parameter(
            0.01 * torch.randn(N_QUBITS * 2, dtype=torch.float64)
        )

        # 🔥 DECODER YANG BENAR
        self.decoder = nn.Sequential(
            nn.Linear(N_QUBITS, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, N_FEATURES)
        )

    def forward(self, x):
        latents = []
        for i in range(x.shape[0]):
            z = torch.stack(quantum_encoder(x[i], self.q_weights))
            latents.append(z)

        z = torch.stack(latents)
        recon = self.decoder(z)
        return recon

# =====================================================
# 6. LOSS FUNCTION (ANOMALY-AWARE)
# =====================================================
class WeightedMSE(nn.Module):
    def forward(self, recon, x):
        error = (recon - x) ** 2
        weights = torch.where(x > x.mean(), 2.0, 1.0)
        return torch.mean(weights * error)

# =====================================================
# 7. TRAINING SETUP
# =====================================================
model = HybridQAE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = WeightedMSE()

best_val = np.inf
patience_cnt = 0

print("\n🚀 TRAINING STARTED\n")

# =====================================================
# 8. TRAIN LOOP + EARLY STOPPING
# =====================================================
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(X_train))
    train_loss = 0

    for i in range(0, len(X_train), BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        batch = X_train[idx]

        optimizer.zero_grad()
        recon = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(X_train) // BATCH_SIZE

    # VALIDATION
    model.eval()
    with torch.no_grad():
        val_recon = model(X_val)
        val_loss = criterion(val_recon, X_val).item()

    print(f"Epoch {epoch+1:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    # EARLY STOP
    if val_loss < best_val:
        best_val = val_loss
        patience_cnt = 0
        torch.save(model.state_dict(), "qae_hybrid_best.pt")
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print("⏹ Early stopping triggered")
            break

print("\n✅ TRAINING FINISHED")
