"""
Leiden clustering pipeline — main branch.

Leiden is treated as a first-class algorithm in the cross-branch benchmark.
Sweeps three resolutions (0.25, 0.5, 1.0), reports silhouette, Davies-Bouldin,
and ARI vs ground-truth cell_type for each. Downstream analysis (DGE,
composition, regional) uses Leiden res=0.50 as the chosen resolution.
"""

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import harmonypy as hm
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from load_data import load_h5ad_data

os.makedirs("figures", exist_ok=True)

sc.settings.verbosity = 3
sc.settings.autoshow = False
sc.settings.set_figure_params(dpi=150, facecolor='white')


class _DropSaveFigMsg(logging.Filter):
    def filter(self, record):
        return "saving figure to file" not in record.getMessage()


sc.settings._root_logger.addFilter(_DropSaveFigMsg())


# ─────────────────────────────────────────────────────────────────────────────
# Read in the data
# ─────────────────────────────────────────────────────────────────────────────
# Random subsample to 50K cells (seeded for reproducibility)
adata = load_h5ad_data('703771a1-236f-4eda-9c04-318d882e149b.h5ad', n_cells=50000)


## 1. Normalize / Preprocess

# Annotate gene populations using readable gene symbols
adata.var["mt"]   = adata.var["feature_name"].str.upper().str.startswith("MT-")
adata.var["ribo"] = adata.var["feature_name"].str.upper().str.startswith(("RPS", "RPL"))
adata.var["hb"]   = adata.var["feature_name"].str.upper().str.contains("^HB[^(P)]")

# Calculate QC metrics
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

# Visualize QC metrics
sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
    save="_qc_metrics.png",
)

# Filter cells — 200 
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Doublet Detection 
sc.pp.scrublet(adata, batch_key="batch")
adata = adata[~adata.obs["predicted_doublet"]].copy()

# Normalization and Feature Selection
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# Identify highly variable genes from raw counts (seurat_v3 expects integer counts).
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,
    batch_key="batch",
    subset=False,
    flavor="seurat_v3",
    layer="counts",
)

adata = adata[:, adata.var["highly_variable"]].copy()


## 2. Batch Effect Correction

# Dimensionality reduction using PCA
sc.tl.pca(adata, svd_solver="arpack")

# PCA variance plot
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True, save="_pca_variance.png")

Z = adata.obsm["X_pca"]

# Run Harmony batch correction
ho = hm.run_harmony(Z, adata.obs, vars_use=["batch"])
adata.obsm["X_pca_harmony"] = ho.Z_corr.T

# Before Harmony: PCA coloured by batch
sc.pl.pca(
    adata,
    color="batch",
    title="PCA before Harmony (coloured by batch)",
    save="_pca_before_harmony.png",
)

# After Harmony: embedding coloured by batch
sc.pl.embedding(
    adata,
    basis="X_pca_harmony",
    color="batch",
    title="PCA after Harmony (coloured by batch)",
    save="_pca_after_harmony.png",
)

# Construct neighborhood graph on Harmony-corrected PCA
sc.pp.neighbors(adata, use_rep="X_pca_harmony")
sc.tl.umap(adata)

# UMAP coloured by batch — sanity check for batch effect removal
sc.pl.umap(
    adata,
    color="batch",
    size=2,
    title="UMAP coloured by batch (after Harmony)",
    save="_umap_batch.png",
)

# UMAP coloured by cell type 
sc.pl.umap(
    adata,
    color="cell_type",
    title="UMAP coloured by cell type",
    save="_umap_cell_type.png",
)


## 3. Clustering: Leiden at three resolutions

print("Running Leiden clustering at three resolutions...")
sc.tl.leiden(adata, resolution=0.25, key_added="leiden_res_0.25", flavor="igraph", directed=False, n_iterations=2)
sc.tl.leiden(adata, resolution=0.5,  key_added="leiden_res_0.50", flavor="igraph", directed=False, n_iterations=2)
sc.tl.leiden(adata, resolution=1.0,  key_added="leiden_res_1.00", flavor="igraph", directed=False, n_iterations=2)

# Side-by-side UMAP of the three resolutions
sc.pl.umap(
    adata,
    color=["leiden_res_0.25", "leiden_res_0.50", "leiden_res_1.00"],
    wspace=0.4,
    title=["Leiden (Res=0.25)", "Leiden (Res=0.50)", "Leiden (Res=1.0)"],
    save="_leiden_comparison.png",
)


## 4. Quantitative Benchmark Metrics
# Silhouette + Davies-Bouldin + ARI vs ground-truth cell_type for each Leiden resolution.

print("\nComputing Leiden benchmark metrics (silhouette, Davies-Bouldin, ARI vs truth)...")
embed = adata.obsm["X_pca_harmony"]
truth_ref = adata.obs["cell_type"].astype(str).astype("category").cat.codes.values

bench_rows = []
for res_str in ["0.25", "0.50", "1.00"]:
    col = f"leiden_res_{res_str}"
    labels = adata.obs[col].astype(int).values
    n_clusters = len(set(labels))
    sample_size = min(5000, len(labels))
    sil = silhouette_score(embed, labels, sample_size=sample_size)
    db = davies_bouldin_score(embed, labels)
    ari_truth = adjusted_rand_score(truth_ref, labels)
    bench_rows.append({
        "method": f"Leiden (res={res_str})",
        "n_clusters": n_clusters,
        "silhouette": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "ari_vs_truth": round(ari_truth, 4),
    })

bench_df = pd.DataFrame(bench_rows)
print("\n=== Benchmark Metrics ===")
print(bench_df.to_string(index=False))

# 3-panel bar chart, best value highlighted in orange.
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
metric_specs = [
    ("silhouette", "Silhouette score\n(higher = better)", "max"),
    ("davies_bouldin", "Davies-Bouldin index\n(lower = better)", "min"),
    ("ari_vs_truth", "ARI vs ground truth\n(higher = better)", "max"),
]
methods = bench_df["method"].tolist()
for ax, (metric, title, best_dir) in zip(axes, metric_specs):
    values = bench_df[metric].values
    bars = ax.bar(methods, values, color="#4292c6")
    best_idx = int(values.argmax() if best_dir == "max" else values.argmin())
    bars[best_idx].set_color("#fd8d3c")
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Leiden — clustering benchmark across resolutions", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("figures/leiden_benchmark.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved figures/leiden_benchmark.png")


## 5. Cluster Interpretation — DGE on the chosen Leiden resolution

print("\nRunning Differential Gene Expression on Leiden res=0.50...")
chosen_cluster_key = "leiden_res_0.50"

sc.tl.rank_genes_groups(
    adata,
    groupby=chosen_cluster_key,
    method="wilcoxon",
    use_raw=False,
)

sc.tl.dendrogram(adata, groupby=chosen_cluster_key, use_rep="X_pca_harmony")

sc.pl.rank_genes_groups_dotplot(
    adata,
    n_genes=5,
    groupby=chosen_cluster_key,
    gene_symbols="feature_name",
    standard_scale="var",
    title="Top 5 Marker Genes per Leiden Cluster",
    save="marker_genes.png",
)

# Print top 10 markers for cluster 0 with readable names
cluster_0_markers = sc.get.rank_genes_groups_df(adata, group="0")
cluster_0_markers["gene_symbol"] = (
    adata.var.loc[cluster_0_markers["names"], "feature_name"].values
)

print("\n--- Top 10 marker genes for Cluster 0 ---")
print(cluster_0_markers[["gene_symbol", "names", "scores", "logfoldchanges", "pvals_adj"]].head(10))


## 6. Cell Type Composition: Anterior vs Posterior Hippocampus

CLUSTER_COL = "leiden_res_0.50"

print("\nGenerating composition plot...")
composition = pd.crosstab(adata.obs["group"], adata.obs[CLUSTER_COL], normalize="index")
ax = composition.plot(kind="bar", stacked=True, figsize=(10, 6))
plt.title("Cell Type Composition: Anterior vs Posterior Hippocampus")
plt.xlabel("Region (group)")
plt.ylabel("Proportion of Cells")
plt.legend(title="Leiden Cluster (res=0.50)",
           bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("figures/composition_anterior_vs_posterior.png", dpi=150, bbox_inches="tight")
plt.close()


## 7. UMAP separated by Region
print("Generating comparative UMAPs...")
sc.pl.umap(adata, color=[CLUSTER_COL, "group"], wspace=0.4, save="_leiden_vs_region.png")


## 8. Gene Expression by Cluster AND Region
# Reuses the Leiden DGE results from Section 5 (must run before Section 9, which overwrites them).
print("Generating comparative dotplot...")
adata.obs["cluster_region"] = (
    adata.obs[CLUSTER_COL].astype(str) + "_" + adata.obs["group"].astype(str)
)

dge_result = adata.uns["rank_genes_groups"]["names"]
top_markers = []
for cluster_name in dge_result.dtype.names:
    top_markers.extend(dge_result[cluster_name][:5])
top_markers = list(dict.fromkeys(top_markers))  # deduplicate, preserve order

marker_symbols = adata.var.loc[top_markers, "feature_name"].tolist()

sc.pl.dotplot(
    adata,
    var_names=marker_symbols,
    groupby="cluster_region",
    gene_symbols="feature_name",
    standard_scale="var",
    title=f"Top Auto-Discovered Markers by Cluster and Region ({CLUSTER_COL})",
    save="marker_by_cluster_region.png",
)


## 9. Global DGE: Anterior vs Posterior
# Overwrites adata.uns['rank_genes_groups'] with the region-keyed DGE.
print("\nRunning DGE between anterior and posterior regions...")
sc.tl.rank_genes_groups(adata, groupby="group", method="wilcoxon", use_raw=False)

sc.pl.rank_genes_groups_dotplot(
    adata,
    n_genes=25,
    gene_symbols="feature_name",
    title="Top Regional Differences: Anterior vs Posterior",
    save="anterior_vs_posterior_dge.png",
)

print("\nLeiden pipeline complete.")
