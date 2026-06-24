import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import umap

# ── Configuración global de fuentes (penalización si < 14) ──────────────────
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

SEED = 42
SPECIES_NAMES = {10: "L. discodactylus", 12: "O. taurinus",
                 17: "C. lineata", 18: "S. grossus", 23: "P. chrysopeplus"}
PALETTE = {10: "#4E79A7", 12: "#F28E2B", 17: "#59A14F",
           18: "#E15759", 23: "#B07AA1"}

# Usamos esos nombres porque renombramos los dataset a algo mas corto.
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
df    = pd.concat([train, test], ignore_index=True)

MEL_COLS = [f"mel_{i}" for i in range(64)]
X = df[MEL_COLS].values
y = df["species_id"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Dataset: {X_scaled.shape[0]} obs · {X_scaled.shape[1]} features")
print(f"Clases: {np.unique(y)}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3.2 — Exploración geométrica y reducción de dimensionalidad
# ═══════════════════════════════════════════════════════════════════════════════

# ── 3.2.1  PCA ──────────────────────────────────────────────────────────────
t0 = time.time()
pca = PCA(random_state=SEED)
X_pca_full = pca.fit_transform(X_scaled)
t_pca = time.time() - t0

X_pca2 = X_pca_full[:, :2]
X_pca3 = X_pca_full[:, :3]

var_acum = np.cumsum(pca.explained_variance_ratio_)
n_95 = int(np.searchsorted(var_acum, 0.95)) + 1
var_2pc = var_acum[1] * 100

print(f"[PCA] tiempo={t_pca:.3f}s  |  var(2 PCs)={var_2pc:.1f}%  |  PCs para 95%={n_95}")

# Figura 1 — Varianza acumulada
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

# ── 3.2.2  t-SNE ────────────────────────────────────────────────────────────
# Usamos X_pca con 30 dims como input (acelera t-SNE y reduce ruido)
X_pre = X_pca_full[:, :min(30, X_pca_full.shape[1])]

t0 = time.time()
tsne = TSNE(n_components=2, perplexity=min(30, len(X_pre) - 1),
            max_iter=1000, random_state=SEED)
X_tsne = tsne.fit_transform(X_pre)
t_tsne = time.time() - t0
print(f"[t-SNE] tiempo={t_tsne:.3f}s  |  KL-div={tsne.kl_divergence_:.4f}")

# ── 3.2.3  UMAP ─────────────────────────────────────────────────────────────
t0 = time.time()
reducer = umap.UMAP(n_components=2, n_neighbors=min(15, len(X_pre) - 1),
                    min_dist=0.1, random_state=SEED)
X_umap = reducer.fit_transform(X_scaled)
t_umap = time.time() - t0
print(f"[UMAP] tiempo={t_umap:.3f}s")

# ── Figura 2 — Proyecciones 2D comparativas ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
datasets_2d = [("PCA", X_pca2), ("t-SNE", X_tsne), ("UMAP", X_umap)]

for ax, (name, X2) in zip(axes, datasets_2d):
    for sp, label in SPECIES_NAMES.items():
        mask = y == sp
        if mask.sum() == 0:
            continue
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   color=PALETTE[sp], label=label, s=60, alpha=0.8, edgecolors="w", lw=0.5)
    ax.set_title(f"Figura 2{['a','b','c'][datasets_2d.index((name,X2))]}. {name}")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(loc="best", fontsize=12, markerscale=1.2)

fig.suptitle("Figura 2. Proyecciones 2D: PCA vs t-SNE vs UMAP", fontsize=15)
fig.tight_layout()
fig.savefig("fig2_proyecciones_2d.png", dpi=150)
plt.close()

# ── Tabla comparativa 3.2 ────────────────────────────────────────────────────
tabla_32 = pd.DataFrame({
    "Método":       ["PCA", "t-SNE", "UMAP"],
    "Tiempo (s)":   [round(t_pca, 3), round(t_tsne, 3), round(t_umap, 3)],
    "Var/Métrica":  [f"{var_2pc:.1f}% var", f"KL={tsne.kl_divergence_:.3f}", "dist local"],
    "Tipo":         ["Lineal", "No lineal", "No lineal"],
})
print("\n── Tabla comparativa 3.2 ──")
print(tabla_32.to_string(index=False))
tabla_32.to_csv("tabla_32_comparativa.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3.3 — Minería de patrones y clustering
# ═══════════════════════════════════════════════════════════════════════════════
# Usamos la proyección UMAP como espacio de trabajo (preserva estructura local)
X_cl = X_umap  # 2D, ya escalado en origen

# ── 3.3.1  DBSCAN ───────────────────────────────────────────────────────────
# Hiperparámetros: eps via curva k-distancias, min_samples=2*dims
from sklearn.neighbors import NearestNeighbors

k = 4  # 2 * n_dims (2D → 4)
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

# Codo via segunda derivada sobre k-distancias ordenadas ascendente
k_dists_asc = np.sort(dists[:, -1])
d2 = np.diff(k_dists_asc, n=2)
elbow_idx = int(np.argmax(d2))
eps_val = float(k_dists_asc[elbow_idx])
eps_val = max(eps_val, 0.3)  # piso minimo para evitar eps degenerados
dbscan = DBSCAN(eps=eps_val, min_samples=k)
labels_db = dbscan.fit_predict(X_cl)
n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise_db = (labels_db == -1).sum()

print(f"\n[DBSCAN] eps={eps_val:.3f}  min_samples={k}")
print(f"  Clústeres: {n_clusters_db}  |  Ruido: {n_noise_db} pts")

# ── 3.3.2  GMM ──────────────────────────────────────────────────────────────
# Selección de K via BIC + Silhouette
K_range = range(2, min(8, len(X_cl)))
bic_vals, sil_vals = [], []

for k_gmm in K_range:
    gmm_tmp = GaussianMixture(n_components=k_gmm, covariance_type="full",
                               random_state=SEED, n_init=3)
    labels_tmp = gmm_tmp.fit_predict(X_cl)
    bic_vals.append(gmm_tmp.bic(X_cl))
    if len(set(labels_tmp)) > 1:
        sil_vals.append(silhouette_score(X_cl, labels_tmp))
    else:
        sil_vals.append(-1)

best_k = list(K_range)[int(np.argmin(bic_vals))]
print(f"[GMM] K óptimo (BIC): {best_k}")

gmm = GaussianMixture(n_components=best_k, covariance_type="full",
                       random_state=SEED, n_init=5)
labels_gmm = gmm.fit_predict(X_cl)

# Figura 4 — BIC y Silhouette vs K
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(list(K_range), bic_vals, marker="o", color="#4E79A7", lw=2)
ax1.axvline(best_k, color="red", ls="--", lw=1.2, label=f"K={best_k}")
ax1.set_xlabel("Número de componentes (K)")
ax1.set_ylabel("BIC")
ax1.set_title("Figura 4a. BIC vs K (GMM)")
ax1.legend()

ax2.plot(list(K_range), sil_vals, marker="s", color="#59A14F", lw=2)
ax2.axvline(best_k, color="red", ls="--", lw=1.2, label=f"K={best_k}")
ax2.set_xlabel("Número de componentes (K)")
ax2.set_ylabel("Silhouette score")
ax2.set_title("Figura 4b. Silhouette vs K (GMM)")
ax2.legend()

fig.tight_layout()
fig.savefig("fig4_gmm_metricas.png", dpi=150)
plt.close()

# ── Figura 5 — Resultados clustering ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# DBSCAN
cmap_db = plt.colormaps["tab10"].resampled(n_clusters_db + 1)
for lbl in sorted(set(labels_db)):
    mask = labels_db == lbl
    color = "black" if lbl == -1 else cmap_db(lbl)
    name  = "Ruido" if lbl == -1 else f"Cluster {lbl}"
    axes[0].scatter(X_cl[mask, 0], X_cl[mask, 1],
                    color=color, label=name, s=60, alpha=0.8, edgecolors="w", lw=0.4)
axes[0].set_title(f"Figura 5a. DBSCAN (eps={eps_val:.2f}, {n_clusters_db} clústers)")
axes[0].set_xlabel("UMAP Dim 1")
axes[0].set_ylabel("UMAP Dim 2")
axes[0].legend(fontsize=12)

# GMM
cmap_gmm = plt.colormaps["tab10"].resampled(best_k)
for lbl in range(best_k):
    mask = labels_gmm == lbl
    axes[1].scatter(X_cl[mask, 0], X_cl[mask, 1],
                    color=cmap_gmm(lbl), label=f"Cluster {lbl}", s=60, alpha=0.8,
                    edgecolors="w", lw=0.4)
axes[1].set_title(f"Figura 5b. GMM (K={best_k} componentes)")
axes[1].set_xlabel("UMAP Dim 1")
axes[1].set_ylabel("UMAP Dim 2")
axes[1].legend(fontsize=12)

fig.suptitle("Figura 5. Resultados de clustering en espacio UMAP", fontsize=15)
fig.tight_layout()
fig.savefig("fig5_clustering.png", dpi=150)
plt.close()

# ── Tabla de métricas de validación 3.3 ──────────────────────────────────────
def metricas(X, labels, nombre):
    mask = labels != -1
    X_v, L_v = X[mask], labels[mask]
    if len(set(L_v)) < 2:
        return {"Método": nombre, "Silhouette": "N/A",
                "Davies-Bouldin": "N/A", "Calinski-Harabasz": "N/A"}
    return {
        "Método":             nombre,
        "Silhouette":         round(silhouette_score(X_v, L_v), 4),
        "Davies-Bouldin":     round(davies_bouldin_score(X_v, L_v), 4),
        "Calinski-Harabasz":  round(calinski_harabasz_score(X_v, L_v), 2),
    }

tabla_33 = pd.DataFrame([
    metricas(X_cl, labels_db, f"DBSCAN (eps={eps_val:.2f})"),
    metricas(X_cl, labels_gmm, f"GMM (K={best_k})"),
])
print("\n── Tabla de métricas 3.3 ──")
print(tabla_33.to_string(index=False))
tabla_33.to_csv("tabla_33_metricas.csv", index=False)

print("\nScript completado. Figuras y tablas guardadas.")