"""
SC3 — Single-Cell Consensus Clustering (Python re-implementation)
=================================================================
Original algorithm: Kiselev et al., Nature Methods 2017
  https://doi.org/10.1038/nmeth.4236

IMPLEMENTATION NOTES
--------------------
- Distances are computed on adata.obsm[basis] (Harmony PCA, 50 dims) rather
  than the raw gene matrix.  High-dimensional gene space (2000 genes) causes
  all pairwise distances to converge, making the Laplacian uninformative.
  The PCA embedding already captures the main variance structure and is the
  same representation used by Leiden, giving a fair comparison.

- Hierarchical clustering uses sklearn AgglomerativeClustering(n_clusters=k)
  which *guarantees* exactly k clusters.  scipy's fcluster with 'maxclust'
  can return fewer than k when the consensus matrix has flat dendrogram regions.

SCALABILITY
-----------
SC3 is O(N^2) in memory.  We subsample n_subsample cells, run full SC3 on
those, then use kNN label transfer to assign the rest.

ALGORITHM (on the subsample)
-----------------------------
1. Extract PCA embedding for subsample  (n_sub x n_pcs).
2. Compute three distance matrices: Euclidean, Pearson, Spearman.
3. For each distance matrix and each k:
     a. Gaussian affinity kernel → normalised graph Laplacian.
     b. Top-k eigenvectors of the Laplacian.
     c. k-means on those eigenvectors (n_init_kmeans restarts).
4. For each k: average binary co-cluster matrices → consensus matrix C.
5. AgglomerativeClustering(n_clusters=k, metric='precomputed') on (1-C).
6. kNN label transfer to all remaining cells.

OUTPUTS
-------
adata.obs['sc3_k{k}']  for each k in k_range.
adata.uns['sc3_params'] dict of parameters used.
"""

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Distance matrices  (computed on the PCA embedding)
# ---------------------------------------------------------------------------

def _euclidean_distance(X: np.ndarray) -> np.ndarray:
    """N×N Euclidean distance matrix, normalised to [0, 1]."""
    D = cdist(X, X, metric='euclidean')
    dmax = D.max()
    return D / dmax if dmax > 0 else D


def _pearson_distance(X: np.ndarray) -> np.ndarray:
    """N×N Pearson correlation distance (1 - r), mapped to [0, 1]."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        R = np.corrcoef(X)
    return np.clip((1.0 - R) / 2.0, 0, 1)


def _spearman_distance(X: np.ndarray) -> np.ndarray:
    """N×N Spearman correlation distance (1 - rho), mapped to [0, 1]."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, _ = spearmanr(X, axis=1)
    if np.ndim(rho) == 0:
        rho = np.array([[1.0, float(rho)], [float(rho), 1.0]])
    return np.clip((1.0 - rho) / 2.0, 0, 1)


# ---------------------------------------------------------------------------
# Graph Laplacian eigenvectors
# ---------------------------------------------------------------------------

def _laplacian_eigenvectors(D: np.ndarray, k: int) -> np.ndarray:
    """
    Distance matrix → normalised graph Laplacian → top-k eigenvectors.

    Affinity kernel: Gaussian with width = median(D[D > 0]).
    Drops the trivial first eigenvector (eigenvalue ≈ 0).
    """
    n = D.shape[0]
    sigma = np.median(D[D > 0]) if (D > 0).any() else 1.0
    A = np.exp(-(D ** 2) / (sigma ** 2))
    np.fill_diagonal(A, 0)

    deg = A.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    L = np.eye(n) - (deg_inv_sqrt[:, None] * A * deg_inv_sqrt[None, :])

    k_safe = min(k + 1, n - 1)
    _, vecs = eigh(L, subset_by_index=[0, k_safe])
    return vecs[:, 1:k + 1]   # skip the trivial first eigenvector


def _kmeans_on_eigvecs(evecs: np.ndarray, k: int,
                        n_init: int = 1, random_state: int = 0) -> np.ndarray:
    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state, max_iter=500)
    return km.fit_predict(evecs)


# ---------------------------------------------------------------------------
# Consensus matrix  →  cluster labels
# ---------------------------------------------------------------------------

def _consensus_matrix(all_labels: list, n_cells: int) -> np.ndarray:
    """
    C[i,j] = fraction of k-means runs in which i and j were co-clustered.
    """
    C = np.zeros((n_cells, n_cells), dtype=np.float32)
    for labels in all_labels:
        C += (labels[:, None] == labels[None, :]).astype(np.float32)
    C /= len(all_labels)
    return C


def _cluster_from_consensus(C: np.ndarray, k: int) -> np.ndarray:
    """
    Agglomerative (complete linkage) clustering on the dissimilarity (1 - C).
    Uses AgglomerativeClustering which *guarantees* exactly k clusters,
    unlike scipy fcluster which can return fewer in degenerate cases.
    Returns 0-indexed integer label array.
    """
    dissim = (1.0 - C).astype(np.float64)
    # Ensure symmetry and zero diagonal (numerical safety)
    dissim = (dissim + dissim.T) / 2.0
    np.fill_diagonal(dissim, 0.0)

    clustering = AgglomerativeClustering(
        n_clusters=k,
        metric='precomputed',
        linkage='complete',
    )
    return clustering.fit_predict(dissim)


# ---------------------------------------------------------------------------
# kNN label transfer to the full dataset
# ---------------------------------------------------------------------------

def _knn_label_transfer(labels_sub: np.ndarray,
                         embed_sub: np.ndarray,
                         embed_full: np.ndarray,
                         subsample_idx: np.ndarray,
                         n_neighbors: int = 15) -> np.ndarray:
    """
    Assign cluster labels to cells not in the subsample via majority-vote kNN.
    """
    n_full = embed_full.shape[0]
    labels_full = np.full(n_full, -1, dtype=int)
    labels_full[subsample_idx] = labels_sub

    mask = np.zeros(n_full, dtype=bool)
    mask[subsample_idx] = True
    remaining_idx = np.where(~mask)[0]

    if len(remaining_idx) == 0:
        return labels_full

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', n_jobs=-1)
    nn.fit(embed_sub)
    _, indices = nn.kneighbors(embed_full[remaining_idx])

    n_clusters = labels_sub.max() + 1
    for i, nbr_idx in enumerate(indices):
        counts = np.bincount(labels_sub[nbr_idx], minlength=n_clusters)
        labels_full[remaining_idx[i]] = counts.argmax()

    return labels_full


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sc3_cluster(
    adata,
    k_range,
    n_subsample: int = 2000,
    basis: str = 'X_pca_harmony',
    n_init_kmeans: int = 10,
    random_state: int = 42,
    verbose: bool = True,
) -> None:
    """
    Run SC3 clustering and store results in adata.obs['sc3_k{k}'].

    Parameters
    ----------
    adata        : AnnData — must have adata.obsm[basis] (run sc.pp.neighbors first).
    k_range      : iterable of int — cluster counts to test, e.g. range(4, 8).
    n_subsample  : int — cells used for SC3 distance computation (keep <= 2000).
    basis        : obsm key for PCA embedding used for distances and kNN transfer.
    n_init_kmeans: int — k-means restarts per Laplacian (10 matches the SC3 paper).
    random_state : int — global random seed.
    verbose      : bool — print progress.
    """
    np.random.seed(random_state)
    k_list = list(k_range)
    n_cells = adata.n_obs
    t0 = time.time()

    # ── 1. Get the embedding ──────────────────────────────────────────────
    if basis not in adata.obsm:
        raise KeyError(f"'{basis}' not found in adata.obsm. "
                       f"Run sc.pp.neighbors(adata, use_rep='{basis}') first.")
    embed_full = adata.obsm[basis]

    # ── 2. Subsample ──────────────────────────────────────────────────────
    if n_subsample < n_cells:
        if verbose:
            print(f"[SC3] Subsampling {n_subsample} / {n_cells} cells.")
        rng = np.random.default_rng(random_state)
        subsample_idx = np.sort(rng.choice(n_cells, size=n_subsample, replace=False))
    else:
        if verbose:
            print(f"[SC3] Using all {n_cells} cells (no subsampling).")
        subsample_idx = np.arange(n_cells)

    embed_sub = embed_full[subsample_idx]
    n_sub = embed_sub.shape[0]

    # ── 3. Distance matrices ──────────────────────────────────────────────
    if verbose:
        print(f"[SC3] Computing distance matrices on {n_sub} cells × "
              f"{embed_sub.shape[1]} PCA dims ...")
    D_euclid   = _euclidean_distance(embed_sub)
    if verbose: print("[SC3]   Euclidean done.")
    D_pearson  = _pearson_distance(embed_sub)
    if verbose: print("[SC3]   Pearson done.")
    D_spearman = _spearman_distance(embed_sub)
    if verbose: print("[SC3]   Spearman done.")

    distance_matrices = [D_euclid, D_pearson, D_spearman]

    # ── 4. For each k: Laplacians → k-means → consensus → labels ─────────
    results = {}

    for k in k_list:
        if verbose:
            print(f"[SC3] Processing k={k} ...")
        all_labels = []

        for D in distance_matrices:
            evecs = _laplacian_eigenvectors(D, k)
            for seed_offset in range(n_init_kmeans):
                lbl = _kmeans_on_eigvecs(
                    evecs, k, n_init=1, random_state=random_state + seed_offset
                )
                all_labels.append(lbl)

        C = _consensus_matrix(all_labels, n_sub)
        # AgglomerativeClustering guarantees exactly k clusters
        sub_labels = _cluster_from_consensus(C, k)
        n_unique = len(np.unique(sub_labels))
        results[k] = sub_labels

        if verbose:
            print(f"[SC3]   k={k}: {n_unique} clusters found in subsample.")

    # ── 5. kNN label transfer ─────────────────────────────────────────────
    for k, sub_labels in results.items():
        if verbose:
            print(f"[SC3] Transferring labels for k={k} to all {n_cells} cells ...")

        if len(subsample_idx) < n_cells:
            full_labels = _knn_label_transfer(
                labels_sub=sub_labels,
                embed_sub=embed_sub,
                embed_full=embed_full,
                subsample_idx=subsample_idx,
            )
        else:
            full_labels = sub_labels

        adata.obs[f'sc3_k{k}'] = pd.Categorical(full_labels.astype(str))

    # ── 6. Metadata ───────────────────────────────────────────────────────
    adata.uns['sc3_params'] = {
        'k_range': k_list,
        'n_subsample': n_subsample,
        'basis': basis,
        'n_init_kmeans': n_init_kmeans,
        'random_state': random_state,
        'distance_space': 'PCA embedding',
    }
    adata.uns['sc3_runtime_s'] = round(time.time() - t0, 1)

    if verbose:
        print(f"[SC3] Done in {adata.uns['sc3_runtime_s']} s.")
        print(f"[SC3] Labels stored in: {[f'sc3_k{k}' for k in k_list]}")


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def sc3_benchmark_metrics(adata, k_range, leiden_key: str = 'leiden_res_0.50',
                            basis: str = 'X_pca_harmony') -> pd.DataFrame:
    """
    Compute clustering quality metrics for each SC3 k and for Leiden.

    Metrics
    -------
    n_clusters    : number of unique clusters found.
    silhouette    : higher is better (max 1.0).
    davies_bouldin: lower is better.
    ari_vs_leiden : Adjusted Rand Index vs Leiden (1.0 = identical).
    """
    embed = adata.obsm[basis]
    rows = []
    leiden_lbl = None

    # Leiden reference row
    if leiden_key in adata.obs.columns:
        leiden_lbl = adata.obs[leiden_key].astype(int).values
        n_leiden = len(np.unique(leiden_lbl))
        sample_size = min(5000, len(leiden_lbl))
        sil = silhouette_score(embed, leiden_lbl, sample_size=sample_size)
        db  = davies_bouldin_score(embed, leiden_lbl)
        rows.append({
            'method': f'Leiden ({leiden_key})',
            'n_clusters': n_leiden,
            'silhouette': round(sil, 4),
            'davies_bouldin': round(db, 4),
            'ari_vs_leiden': 1.0,
        })

    # SC3 rows
    for k in list(k_range):
        col = f'sc3_k{k}'
        if col not in adata.obs.columns:
            continue
        sc3_lbl = adata.obs[col].astype(int).values
        n_unique = len(np.unique(sc3_lbl))
        sample_size = min(5000, len(sc3_lbl))
        sil = silhouette_score(embed, sc3_lbl, sample_size=sample_size)
        db  = davies_bouldin_score(embed, sc3_lbl)
        ari = adjusted_rand_score(leiden_lbl, sc3_lbl) if leiden_lbl is not None else np.nan
        rows.append({
            'method': f'SC3 (k={k})',
            'n_clusters': n_unique,
            'silhouette': round(sil, 4),
            'davies_bouldin': round(db, 4),
            'ari_vs_leiden': round(ari, 4),
        })

    return pd.DataFrame(rows)


def sc3_benchmark_plot(adata, k_range, leiden_key: str = 'leiden_res_0.50',
                        basis: str = 'X_pca_harmony', save: str = None):
    """
    Three-panel bar chart: Silhouette score, Davies-Bouldin index, ARI vs Leiden.
    Saves automatically to figures/sc3_benchmark.png.
    """
    df = sc3_benchmark_metrics(adata, k_range, leiden_key=leiden_key, basis=basis)

    print("\n=== Benchmark Metrics ===")
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ['#d95f02' if 'Leiden' in m else '#1b7837' for m in df['method']]
    methods = df['method'].tolist()

    specs = [
        ('silhouette',    'Silhouette score\n(higher = better)',    'max'),
        ('davies_bouldin','Davies-Bouldin index\n(lower = better)', 'min'),
        ('ari_vs_leiden', 'ARI vs Leiden\n(1.0 = identical)',       'max'),
    ]

    for ax, (col, label, better) in zip(axes, specs):
        vals = df[col].values
        bars = ax.bar(range(len(methods)), vals, color=colors,
                      edgecolor='white', linewidth=0.5)

        # Highlight the best bar with an orange border
        best_val = vals.max() if better == 'max' else vals.min()
        for bar, val in zip(bars, vals):
            if np.isclose(val, best_val):
                bar.set_edgecolor('#e6550d')
                bar.set_linewidth(2)

        ax.set_title(label, fontsize=10)
        ax.set_xticks(range(len(methods)))                          # fix for warning
        ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('SC3 vs Leiden — clustering benchmark', fontsize=12, y=1.02)
    plt.tight_layout()

    out_path = save if save else 'figures/sc3_benchmark.png'
    import os
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[SC3] Benchmark plot saved to {out_path}")
    plt.show()

    return df
