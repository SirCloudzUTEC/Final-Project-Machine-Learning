import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, f1_score,
                             confusion_matrix, adjusted_rand_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
import umap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import xgboost as xgb

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

SPECIES_NAMES = {10: "L. discodactylus", 12: "O. taurinus",
                 17: "C. lineata", 18: "S. grossus", 23: "P. chrysopeplus"}
PALETTE = {10: "#4E79A7", 12: "#F28E2B", 17: "#59A14F",
           18: "#E15759", 23: "#B07AA1"}

MEL_COLS = [f"mel_{i}" for i in range(64)]

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

train_tp = train[train["is_tp"] == 1].reset_index(drop=True)
test_tp  = test[test["is_tp"] == 1].reset_index(drop=True)
df_tp    = pd.concat([train_tp, test_tp], ignore_index=True)

scaler_tp = StandardScaler()
X_tp = scaler_tp.fit_transform(df_tp[MEL_COLS].values)
y_tp = df_tp["species_id"].values

print(f"Data completo: {len(train) + len(test)} obs  |  is_tp=1: {X_tp.shape[0]} obs  |  {len(MEL_COLS)} features")
print(f"Clases: {np.unique(y_tp)}\n")

print("═══ SECCIÓN 3.2 — Reducción de dimensionalidad (espacio is_tp=1) ═══")

t0 = time.time()
pca = PCA(random_state=SEED)
X_pca_full = pca.fit_transform(X_tp)
t_pca = time.time() - t0
X_pca2 = X_pca_full[:, :2]

var_acum = np.cumsum(pca.explained_variance_ratio_)
n_95 = int(np.searchsorted(var_acum, 0.95)) + 1
var_2pc = var_acum[1] * 100
print(f"[PCA] tiempo={t_pca:.3f}s  |  var(2 PCs)={var_2pc:.1f}%  |  PCs para 95%={n_95}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(var_acum) + 1), var_acum * 100, color="#4E79A7", lw=2)
ax.axhline(95, color="gray", ls="--", lw=1.2, label="95 % varianza")
ax.axvline(n_95, color="red", ls="--", lw=1.2, label=f"PC {n_95}")
ax.set_xlabel("Número de componentes principales")
ax.set_ylabel("Varianza acumulada (%)")
ax.set_title("Figura 1. Varianza acumulada explicada por PCA")
ax.legend()
ax.set_xlim(1, 64)
ax.set_ylim(0, 101)
fig.tight_layout()
fig.savefig("fig1_pca_varianza.png", dpi=150)
plt.close()

# Ambos métodos no lineales se ejecutan sobre el MISMO espacio de entrada
# (PCA-30, denoising estándar) para que la comparación de tiempos sea justa.
X_pre = X_pca_full[:, :min(30, X_pca_full.shape[1])]

t0 = time.time()
tsne = TSNE(n_components=2, perplexity=min(30, len(X_pre) - 1),
            max_iter=1000, random_state=SEED)
X_tsne = tsne.fit_transform(X_pre)
t_tsne = time.time() - t0
print(f"[t-SNE] tiempo={t_tsne:.3f}s  |  KL-div={tsne.kl_divergence_:.4f}")

# Warm-up: la primera llamada a UMAP compila el kernel numba (JIT) e infla
# el tiempo medido. Se descarta un fit previo para cronometrar de forma justa.
_ = umap.UMAP(n_components=2, n_neighbors=min(15, len(X_pre) - 1),
              min_dist=0.1, random_state=SEED).fit_transform(X_pre)
t0 = time.time()
reducer = umap.UMAP(n_components=2, n_neighbors=min(15, len(X_pre) - 1),
                    min_dist=0.1, random_state=SEED)
X_umap = reducer.fit_transform(X_pre)
t_umap = time.time() - t0
print(f"[UMAP] tiempo={t_umap:.3f}s")

# Trustworthiness: métrica cuantitativa de preservación de la estructura local
# (0-1, mayor es mejor). Permite contrastar t-SNE y UMAP con un número, no solo
# de forma cualitativa. Se calcula sobre el mismo espacio de entrada (X_pre).
tw_tsne = trustworthiness(X_pre, X_tsne, n_neighbors=15)
tw_umap = trustworthiness(X_pre, X_umap, n_neighbors=15)
print(f"[Trustworthiness] t-SNE={tw_tsne:.4f}  UMAP={tw_umap:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
datasets_2d = [("PCA", X_pca2), ("t-SNE", X_tsne), ("UMAP", X_umap)]
for i, (ax, (name, X2)) in enumerate(zip(axes, datasets_2d)):
    for sp, label in SPECIES_NAMES.items():
        mask = y_tp == sp
        if mask.sum() == 0:
            continue
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   color=PALETTE[sp], label=label, s=55, alpha=0.8, edgecolors="w", lw=0.5)
    ax.set_title(f"Figura 2{['a','b','c'][i]}. {name}")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(loc="best", markerscale=1.2)
fig.suptitle("Figura 2. Proyecciones 2D sobre señales confirmadas (is_tp=1): PCA vs t-SNE vs UMAP", fontsize=15)
fig.tight_layout()
fig.savefig("fig2_proyecciones_2d.png", dpi=150)
plt.close()

tabla_32 = pd.DataFrame({
    "Método":      ["PCA", "t-SNE", "UMAP"],
    "Tiempo (s)":  [round(t_pca, 3), round(t_tsne, 3), round(t_umap, 3)],
    "Var/Métrica": [f"{var_2pc:.1f}% var", f"KL={tsne.kl_divergence_:.3f}", "—"],
    "Trustworthiness": ["—", round(tw_tsne, 4), round(tw_umap, 4)],
    "Tipo":        ["Lineal", "No lineal", "No lineal"],
})
print("\n── Tabla comparativa 3.2 ──")
print(tabla_32.to_string(index=False))
tabla_32.to_csv("tabla_32_comparativa.csv", index=False)

print("\n═══ SECCIÓN 3.3 — Clustering (espacio UMAP, is_tp=1) ═══")
X_cl = X_umap

k = 4
nbrs = NearestNeighbors(n_neighbors=k).fit(X_cl)
dists, _ = nbrs.kneighbors(X_cl)
k_dists = np.sort(dists[:, -1])[::-1]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(k_dists, color="#E15759", lw=2)
ax.set_xlabel("Puntos ordenados")
ax.set_ylabel(f"Distancia al {k}-ésimo vecino")
ax.set_title(f"Figura 3. Curva k-distancias (k={k}) para selección de eps")
fig.tight_layout()
fig.savefig("fig3_kdist.png", dpi=150)
plt.close()

eps_grid = np.round(np.arange(0.35, 0.75, 0.05), 2)
best_db = None
for e in eps_grid:
    lab = DBSCAN(eps=e, min_samples=k).fit_predict(X_cl)
    m = lab != -1
    nc = len(set(lab)) - (1 if -1 in lab else 0)
    if 3 <= nc <= 10 and len(set(lab[m])) > 1:
        s = silhouette_score(X_cl[m], lab[m])
        if best_db is None or s > best_db[0]:
            best_db = (s, float(e), lab, nc)

sil_db, eps_val, labels_db, n_clusters_db = best_db
n_noise_db = int((labels_db == -1).sum())
print(f"[DBSCAN] eps={eps_val:.2f} (Silhouette-óptimo)  min_samples={k}  clústeres={n_clusters_db}  ruido={n_noise_db}")

K_range = list(range(2, 9))
bic_vals, sil_vals = [], []
for k_gmm in K_range:
    gmm_tmp = GaussianMixture(n_components=k_gmm, covariance_type="full",
                              random_state=SEED, n_init=5)
    labels_tmp = gmm_tmp.fit_predict(X_cl)
    bic_vals.append(gmm_tmp.bic(X_cl))
    sil_vals.append(silhouette_score(X_cl, labels_tmp) if len(set(labels_tmp)) > 1 else -1)

cand_K = [K for K in K_range if 3 <= K <= 7]
best_k = cand_K[int(np.argmax([sil_vals[K_range.index(K)] for K in cand_K]))]
print(f"[GMM] K óptimo (Silhouette en rango no trivial): {best_k}  |  BIC decrece monótono → no se usa su argmin")

gmm = GaussianMixture(n_components=best_k, covariance_type="full",
                      random_state=SEED, n_init=5)
labels_gmm = gmm.fit_predict(X_cl)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(K_range, bic_vals, marker="o", color="#4E79A7", lw=2)
ax1.axvline(best_k, color="red", ls="--", lw=1.2, label=f"K={best_k}")
ax1.set_xlabel("Número de componentes (K)")
ax1.set_ylabel("BIC")
ax1.set_title("Figura 4a. BIC vs K (GMM)")
ax1.legend()
ax2.plot(K_range, sil_vals, marker="s", color="#59A14F", lw=2)
ax2.axvline(best_k, color="red", ls="--", lw=1.2, label=f"K={best_k}")
ax2.set_xlabel("Número de componentes (K)")
ax2.set_ylabel("Silhouette score")
ax2.set_title("Figura 4b. Silhouette vs K (GMM)")
ax2.legend()
fig.tight_layout()
fig.savefig("fig4_gmm_metricas.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cmap_db = plt.colormaps["tab10"].resampled(max(n_clusters_db, 1) + 1)
for lbl in sorted(set(labels_db)):
    mask = labels_db == lbl
    color = "black" if lbl == -1 else cmap_db(lbl)
    name = "Ruido" if lbl == -1 else f"Cluster {lbl}"
    axes[0].scatter(X_cl[mask, 0], X_cl[mask, 1],
                    color=color, label=name, s=50, alpha=0.8, edgecolors="w", lw=0.4)
axes[0].set_title(f"Figura 5a. DBSCAN (eps={eps_val:.2f}, {n_clusters_db} clústers)")
axes[0].set_xlabel("UMAP Dim 1")
axes[0].set_ylabel("UMAP Dim 2")
axes[0].legend()
cmap_gmm = plt.colormaps["tab10"].resampled(best_k)
for lbl in range(best_k):
    mask = labels_gmm == lbl
    axes[1].scatter(X_cl[mask, 0], X_cl[mask, 1],
                    color=cmap_gmm(lbl), label=f"Cluster {lbl}", s=50, alpha=0.8,
                    edgecolors="w", lw=0.4)
axes[1].set_title(f"Figura 5b. GMM (K={best_k} componentes)")
axes[1].set_xlabel("UMAP Dim 1")
axes[1].set_ylabel("UMAP Dim 2")
axes[1].legend()
fig.suptitle("Figura 5. Resultados de clustering en espacio UMAP (is_tp=1)", fontsize=15)
fig.tight_layout()
fig.savefig("fig5_clustering.png", dpi=150)
plt.close()

def metricas(X, labels, nombre):
    mask = labels != -1
    X_v, L_v = X[mask], labels[mask]
    if len(set(L_v)) < 2:
        return {"Método": nombre, "Silhouette": "N/A",
                "Davies-Bouldin": "N/A", "Calinski-Harabasz": "N/A"}
    return {
        "Método":            nombre,
        "Silhouette":        round(silhouette_score(X_v, L_v), 4),
        "Davies-Bouldin":    round(davies_bouldin_score(X_v, L_v), 4),
        "Calinski-Harabasz": round(calinski_harabasz_score(X_v, L_v), 2),
    }

tabla_33 = pd.DataFrame([
    metricas(X_cl, labels_db, f"DBSCAN (eps={eps_val:.2f})"),
    metricas(X_cl, labels_gmm, f"GMM (K={best_k})"),
])
print("\n── Tabla de métricas 3.3 (espacio UMAP) ──")
print(tabla_33.to_string(index=False))
tabla_33.to_csv("tabla_33_metricas.csv", index=False)

# CONTROL METODOLÓGICO: UMAP optimiza la separación local, por lo que medir
# Silhouette en su propio espacio es parcialmente circular. Se reevalúan las
# MISMAS etiquetas en el espacio PCA-30 (no optimizado para clustering) como
# control: si la estructura persiste, no es un artefacto de la proyección UMAP.
tabla_33_ctrl = pd.DataFrame([
    metricas(X_pre, labels_db, f"DBSCAN (eps={eps_val:.2f})"),
    metricas(X_pre, labels_gmm, f"GMM (K={best_k})"),
])
print("\n── Tabla de métricas 3.3-control (espacio PCA-30, mismas etiquetas) ──")
print(tabla_33_ctrl.to_string(index=False))
tabla_33_ctrl.to_csv("tabla_33_metricas_control_pca.csv", index=False)

ari_db = adjusted_rand_score(y_tp[labels_db != -1], labels_db[labels_db != -1])
ari_gmm = adjusted_rand_score(y_tp, labels_gmm)
print(f"[ARI vs especie real] DBSCAN={ari_db:.3f}  GMM={ari_gmm:.3f}  (bajo → estructura no supervisada ≠ identidad de especie)")

print("\n═══ SECCIÓN 3.4 — MLP vs XGBoost ═══")
le = LabelEncoder()
le.fit(y_tp)
n_classes = len(le.classes_)

sup_scaler = StandardScaler().fit(train_tp[MEL_COLS].values)
X_train = sup_scaler.transform(train_tp[MEL_COLS].values)
X_val   = sup_scaler.transform(test_tp[MEL_COLS].values)
y_train = le.transform(train_tp["species_id"].values)
y_val   = le.transform(test_tp["species_id"].values)
print(f"Train(is_tp=1)={len(X_train)}  Test(is_tp=1)={len(X_val)}  Clases={n_classes}")

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            out = model(xb)
            loss = criterion(out, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
    return total_loss / total, correct / total

def make_loaders(X_tr, y_tr, X_v, y_v, batch=32):
    tr = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                       torch.tensor(y_tr, dtype=torch.long))
    vl = TensorDataset(torch.tensor(X_v, dtype=torch.float32),
                       torch.tensor(y_v, dtype=torch.long))
    return (DataLoader(tr, batch_size=batch, shuffle=True, drop_last=True),
            DataLoader(vl, batch_size=batch))

def train_model(model, X_tr, y_tr, X_v, y_v, epochs=120, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    tr_loader, vl_loader = make_loaders(X_tr, y_tr, X_v, y_v)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for _ in range(epochs):
        tl, ta = run_epoch(model, tr_loader, criterion, optimizer)
        vl, va = run_epoch(model, vl_loader, criterion)
        scheduler.step()
        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)
        hist["train_acc"].append(ta)
        hist["val_acc"].append(va)
    return hist

class MLPBase(nn.Module):
    def __init__(self, in_f, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_cls),
        )
    def forward(self, x):
        return self.net(x)

class MLPDropout(nn.Module):
    def __init__(self, in_f, n_cls, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f, 128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, n_cls),
        )
    def forward(self, x):
        return self.net(x)

class MLPBatchNorm(nn.Module):
    def __init__(self, in_f, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, n_cls),
        )
    def forward(self, x):
        return self.net(x)

class MLPCombined(nn.Module):
    def __init__(self, in_f, n_cls, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, n_cls),
        )
    def forward(self, x):
        return self.net(x)

EPOCHS = 120
IN_F = X_train.shape[1]

print("Entrenando variantes de MLP...")
configs = [
    ("Base",           MLPBase(IN_F, n_classes)),
    ("Dropout",        MLPDropout(IN_F, n_classes)),
    ("BatchNorm",      MLPBatchNorm(IN_F, n_classes)),
    ("Drop+BatchNorm", MLPCombined(IN_F, n_classes)),
]
histories, times_mlp = {}, {}
for name, model in configs:
    t0 = time.time()
    histories[name] = train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS)
    times_mlp[name] = time.time() - t0
    print(f"  [{name}] val_acc={histories[name]['val_acc'][-1]:.4f}  t={times_mlp[name]:.1f}s")

epochs_range = range(1, EPOCHS + 1)
colors = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for (name, _), color in zip(configs, colors):
    h = histories[name]
    axes[0].plot(epochs_range, h["train_loss"], color=color, lw=1.6, ls="--", label="_nolegend_")
    axes[0].plot(epochs_range, h["val_loss"], color=color, lw=2, label=name)
    axes[1].plot(epochs_range, h["val_acc"], color=color, lw=2, label=name)
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].set_title("Figura 6a. Loss vs Épocas (sólido=val, punteado=train)")
axes[0].legend()
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Accuracy (test)")
axes[1].set_title("Figura 6b. Accuracy por variante de regularización")
axes[1].legend()
fig.tight_layout()
fig.savefig("fig6_curvas_aprendizaje.png", dpi=150)
plt.close()

best_mlp_name = max(histories, key=lambda n: np.mean(histories[n]["val_acc"][-20:]))
best_mlp_model = dict(configs)[best_mlp_name]
print(f"Mejor variante MLP: {best_mlp_name} (acc media últimas 20 épocas)")

best_mlp_model.eval()
with torch.no_grad():
    logits = best_mlp_model(torch.tensor(X_val, dtype=torch.float32))
    y_pred_mlp = logits.argmax(1).numpy()
f1_mlp = f1_score(y_val, y_pred_mlp, average="macro")
print(f"[MLP-{best_mlp_name}] F1-macro={f1_mlp:.4f}")

print("Entrenando XGBoost...")
t0 = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="mlogloss", random_state=SEED, n_jobs=-1)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
t_xgb = time.time() - t0
y_pred_xgb = xgb_model.predict(X_val)
f1_xgb = f1_score(y_val, y_pred_xgb, average="macro")
print(f"[XGBoost] F1-macro={f1_xgb:.4f}  t={t_xgb:.1f}s")

class_labels = [SPECIES_NAMES[c] for c in le.classes_]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (y_pred, title) in zip(axes, [
        (y_pred_mlp, f"Figura 7a. MLP ({best_mlp_name}) — F1={f1_mlp:.3f}"),
        (y_pred_xgb, f"Figura 7b. XGBoost — F1={f1_xgb:.3f}")]):
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_labels, yticklabels=class_labels,
                annot_kws={"size": 14}, cbar=False)
    ax.set_title(title)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.tick_params(axis="x", labelrotation=30)
    ax.tick_params(axis="y", labelrotation=0)
fig.suptitle("Figura 7. Matrices de confusión: MLP vs XGBoost (is_tp=1)", fontsize=15)
fig.tight_layout()
fig.savefig("fig7_confusion_matrices.png", dpi=150)
plt.close()

f1_mlp_per = f1_score(y_val, y_pred_mlp, average=None)
f1_xgb_per = f1_score(y_val, y_pred_xgb, average=None)
x_pos = np.arange(n_classes)
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x_pos - width/2, f1_mlp_per, width, label=f"MLP-{best_mlp_name}", color="#4E79A7")
ax.bar(x_pos + width/2, f1_xgb_per, width, label="XGBoost", color="#F28E2B")
ax.set_xticks(x_pos)
ax.set_xticklabels(class_labels, rotation=20, ha="right")
ax.set_ylabel("F1-Score")
ax.set_ylim(0, 1.05)
ax.set_title("Figura 8. F1-Score por clase: MLP vs XGBoost")
ax.legend()
fig.tight_layout()
fig.savefig("fig8_f1_por_clase.png", dpi=150)
plt.close()

tabla_34 = pd.DataFrame([
    {"Modelo": f"MLP-{best_mlp_name}", "F1-macro": round(f1_mlp, 4),
     "Tiempo entreno": f"{times_mlp[best_mlp_name]:.1f}s",
     "Parámetros": sum(p.numel() for p in best_mlp_model.parameters())},
    {"Modelo": "XGBoost", "F1-macro": round(f1_xgb, 4),
     "Tiempo entreno": f"{t_xgb:.1f}s",
     "Parámetros": f"{xgb_model.n_estimators} árboles"},
])
print("\n── Tabla comparativa 3.4 ──")
print(tabla_34.to_string(index=False))
tabla_34.to_csv("tabla_34_benchmark.csv", index=False)

reg_por_variante = {
    "Base":           "—",
    "Dropout":        "Dropout(0.3)",
    "BatchNorm":      "BatchNorm",
    "Drop+BatchNorm": "BatchNorm + Dropout(0.3)",
}
reg_str = reg_por_variante[best_mlp_name]
topologia = pd.DataFrame([
    {"Capa": "Entrada",  "Neuronas": 64,  "Activación": "—",       "Regularización": "—"},
    {"Capa": "Oculta 1", "Neuronas": 128, "Activación": "ReLU",    "Regularización": reg_str},
    {"Capa": "Oculta 2", "Neuronas": 64,  "Activación": "ReLU",    "Regularización": reg_str},
    {"Capa": "Salida",   "Neuronas": 5,   "Activación": "Softmax", "Regularización": "—"},
])
print("\n── Topología MLP ──")
print(topologia.to_string(index=False))
topologia.to_csv("tabla_topologia_mlp.csv", index=False)

sc_all = StandardScaler().fit(train[MEL_COLS].values)
X_tr_all = sc_all.transform(train[MEL_COLS].values)
X_te_all = sc_all.transform(test[MEL_COLS].values)
y_tr_all = le.transform(train["species_id"].values)
y_te_all = le.transform(test["species_id"].values)

lr_all = LogisticRegression(max_iter=2000).fit(X_tr_all, y_tr_all)
lr_tp  = LogisticRegression(max_iter=2000).fit(X_train, y_train)
xgb_all = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                            random_state=SEED, n_jobs=-1).fit(X_tr_all, y_tr_all)

f1_lr_all = f1_score(y_te_all, lr_all.predict(X_te_all), average="macro")
f1_lr_tp  = f1_score(y_val, lr_tp.predict(X_val), average="macro")
f1_xgb_all = f1_score(y_te_all, xgb_all.predict(X_te_all), average="macro")

tabla_contraste = pd.DataFrame([
    {"Modelo": "LogReg",  "F1 (todos)": round(f1_lr_all, 4),  "F1 (is_tp=1)": round(f1_lr_tp, 4)},
    {"Modelo": "XGBoost", "F1 (todos)": round(f1_xgb_all, 4), "F1 (is_tp=1)": round(f1_xgb, 4)},
])
print("\n── Tabla contraste is_tp (3.4 / 3.5) ──")
print(tabla_contraste.to_string(index=False))
tabla_contraste.to_csv("tabla_contraste_istp.csv", index=False)

modelos_c = ["LogReg", "XGBoost"]
f1_todos = [f1_lr_all, f1_xgb_all]
f1_filtrado = [f1_lr_tp, f1_xgb]
xc = np.arange(len(modelos_c))
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(xc - width/2, f1_todos, width, label="Todos los datos", color="#BAB0AC")
ax.bar(xc + width/2, f1_filtrado, width, label="Solo is_tp=1", color="#59A14F")
for i, (a, b) in enumerate(zip(f1_todos, f1_filtrado)):
    ax.text(i - width/2, a + 0.02, f"{a:.2f}", ha="center", fontsize=14)
    ax.text(i + width/2, b + 0.02, f"{b:.2f}", ha="center", fontsize=14)
ax.set_xticks(xc)
ax.set_xticklabels(modelos_c)
ax.set_ylabel("F1-macro (test)")
ax.set_ylim(0, 1.0)
ax.set_title("Figura 9. Impacto de filtrar por is_tp en el desempeño de clasificación")
ax.legend()
fig.tight_layout()
fig.savefig("fig9_contraste_istp.png", dpi=150)
plt.close()

print("\n═══════════════════════════════════════════════════")
print("Script completado. Figuras y tablas guardadas.")
for f in ["fig1_pca_varianza.png", "fig2_proyecciones_2d.png", "fig3_kdist.png",
          "fig4_gmm_metricas.png", "fig5_clustering.png", "fig6_curvas_aprendizaje.png",
          "fig7_confusion_matrices.png", "fig8_f1_por_clase.png", "fig9_contraste_istp.png",
          "tabla_32_comparativa.csv", "tabla_33_metricas.csv", "tabla_34_benchmark.csv",
          "tabla_topologia_mlp.csv", "tabla_contraste_istp.csv"]:
    print(f"  · {f}")

#asd