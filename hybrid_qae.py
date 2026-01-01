# =========================================================
# HYBRID QUANTUM AUTOENCODER - TESTING SCRIPT (FINAL FIXED)
# =========================================================

import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp
import pandas as pd
import numpy as np

print("=" * 60)
print("HYBRID QUANTUM AUTOENCODER - TESTING")
print("=" * 60)

# =========================================================
# 1. CONFIG
# =========================================================
N_FEATURES = 16
N_QUBITS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_FILE = "test_data_qae_processed.csv"
MODEL_FILE = "qae_hybrid_best.pt"
OUTPUT_FILE = "qae_thresholded1_results.csv"

torch.set_default_dtype(torch.float64)

# =========================================================
# 2. LOAD TEST DATA
# =========================================================
df_test = pd.read_csv(TEST_FILE)
X_test = torch.tensor(
    df_test.values,
    dtype=torch.float64,
    device=DEVICE
)

print(f"✓ Test data loaded: {X_test.shape}")

# =========================================================
# 3. QUANTUM DEVICE
# =========================================================
dev = qml.device("default.qubit", wires=N_QUBITS)

# =========================================================
# 4. QUANTUM ENCODER (HARUS IDENTIK DENGAN TRAINING)
# =========================================================
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_encoder(x, weights):

    # Angle encoding (16 → 4 qubit)
    for q in range(N_QUBITS):
        angle = torch.mean(x[q*4:(q+1)*4]) * pnp.pi
        qml.RY(angle, wires=q)

    # Entanglement
    for q in range(N_QUBITS - 1):
        qml.CNOT(wires=[q, q + 1])

    # Variational layer
    idx = 0
    for q in range(N_QUBITS):
        qml.RY(weights[idx], wires=q)
        qml.RZ(weights[idx + 1], wires=q)
        idx += 2

    return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]

# =========================================================
# 5. HYBRID QAE MODEL (SAMA DENGAN TRAINING)
# =========================================================
class HybridQAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.q_weights = nn.Parameter(
            torch.zeros(N_QUBITS * 2, dtype=torch.float64)
        )

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
        return self.decoder(z)

# =========================================================
# 6. LOAD TRAINED MODEL (FIXED, NO KEYERROR)
# =========================================================
model = HybridQAE().to(DEVICE)
model.load_state_dict(
    torch.load(MODEL_FILE, map_location=DEVICE)
)
model.eval()

print("✓ Model loaded successfully")

# =========================================================
# 7. RECONSTRUCTION ERROR
# =========================================================
with torch.no_grad():
    recon = model(X_test)
    mse = torch.mean((recon - X_test) ** 2, dim=1)

mse_np = mse.cpu().numpy()

# =========================================================
# 8. THRESHOLDING (UNSUPERVISED ANOMALY)
# =========================================================
threshold = np.percentile(mse_np, 95)
y_pred = (mse_np > threshold).astype(int)

print(f"✓ Threshold (95%) : {threshold:.6f}")
print(f"✓ Anomaly detected: {y_pred.sum()} / {len(y_pred)}")

# =========================================================
# 9. SAVE RESULTS (SIAP F1-SCORE)
# =========================================================
df_out = pd.DataFrame({
    "reconstruction_error": mse_np,
    "predicted_label": y_pred
})

df_out.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("TESTING FINISHED")
print(f"Output file: {OUTPUT_FILE}")
print("=" * 60)

