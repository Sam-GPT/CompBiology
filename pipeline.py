import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import scanpy.external as sce
import harmonypy as hm
from sklearn.cluster import DBSCAN
import numpy as np

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=150, facecolor='white')


# Read in the data
print("Reading data from 'data.h5ad'...", end='\t')
adata = sc.read_h5ad('data.h5ad')
print("Done")

# Limit to 50K cells
adata = adata[:50000, :].copy()


## 1. Normalize / Preprocess

# Annotate gene populations
print("Annotating gene populations...")
adata.var["mt"] = adata.var["feature_name"].str.upper().str.startswith("MT-")
print("\tMitochondrial genes")
adata.var["ribo"] = adata.var["feature_name"].str.upper().str.startswith(("RPS", "RPL"))
print("\tRibosomal genes")
adata.var["hb"] = adata.var["feature_name"].str.upper().str.contains("^HB[^(P)]")
print("\tHemoglobin genes")

# Calculate QC metrics
print("Calculating QC metrics")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

# Filter cells
print("Filtering cells")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Doublet Detection
print("Doublet detection")
sc.pp.scrublet(adata, batch_key="batch")

adata = adata[adata.obs["doublet_score"] < 0.56].copy()

# Normalization and Feature Selection
adata.layers["counts"] = adata.X.copy()

print("Normalizing to median total counts")
sc.pp.normalize_total(adata)
print("Logarithmize the data")
sc.pp.log1p(adata)

print("Identifying HVGs (highly variable genes)")
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
print("Batch Effect Correction")

print("\tDimensionality reduction")
sc.tl.pca(adata, svd_solver="arpack")

Z = adata.obsm["X_pca"]

ho = hm.run_harmony(Z, adata.obs, vars_use=["batch"])
adata.obsm["X_pca_harmony"] = ho.Z_corr.T

print("\tConstructing neighbourhood graph using Harmony-corrected PCA embeddings")
sc.pp.neighbors(adata, use_rep="X_pca_harmony")
sc.tl.umap(adata)


## 3. Clustering: DBSCAN
print("Running DBSCAN clustering...")

# Run on the 2D UMAP embedding rather than PCA
# the UMAP separates the visible lobes cleanly
#  so distances are uniform and a single eps works globally
embedding = adata.obsm["X_umap"]   # shape: (n_cells, 2)

DBSCAN_PARAMS = [
    {"eps": 0.3, "min_samples": 10, "key": "dbscan_eps0.3"},
    {"eps": 0.5, "min_samples": 10, "key": "dbscan_eps0.5"},
    {"eps": 0.8, "min_samples": 10, "key": "dbscan_eps0.8"},
]

for params in DBSCAN_PARAMS:
    print(f"\tRunning DBSCAN (eps={params['eps']}, min_samples={params['min_samples']})...")
    db = DBSCAN(eps=params["eps"], min_samples=params["min_samples"], n_jobs=-1)
    labels = db.fit_predict(embedding)

    str_labels = np.where(labels == -1, "Noise", labels.astype(str))
    adata.obs[params["key"]] = pd.Categorical(str_labels)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"\t  -> {n_clusters} clusters found, {n_noise} noise points ({n_noise/len(labels)*100:.1f}%)")

# Visualize the three parameterisations side by side
sc.pl.umap(
    adata,
    color=[p["key"] for p in DBSCAN_PARAMS],
    wspace=0.4,
    title=[f"DBSCAN (eps={p['eps']})" for p in DBSCAN_PARAMS],
)

# Choose the best parameterisation for downstream analysis
chosen_cluster_key = "dbscan_eps0.5"

# Noise cells confuse rank_genes_groups — work on a clean subset for DGE
adata_clean = adata[adata.obs[chosen_cluster_key] != "Noise"].copy()


## 4. Cluster Interpretation
print("Running Differential Gene Expression to find marker genes...")

sc.tl.rank_genes_groups(
    adata_clean,
    groupby=chosen_cluster_key,
    method="wilcoxon",
    use_raw=False,
)

sc.pl.rank_genes_groups_dotplot(
    adata_clean,
    n_genes=5,
    groupby=chosen_cluster_key,
    standard_scale="var",
    title="Top 5 Marker Genes per Cluster (DBSCAN)",
)

cluster_0_markers = sc.get.rank_genes_groups_df(adata_clean, group="0")
print("\n--- Top 10 marker genes for Cluster 0 ---")
print(cluster_0_markers.head(10))


## 5. Cell Type Composition by Region
print("Generating composition plot...")

# Cross-tabulate region (group) vs DBSCAN cluster
# normalize per row so each bar sums to 1 and regions with
#  different cell counts are directly comparable.
composition = pd.crosstab(
    adata_clean.obs['group'],
    adata_clean.obs[chosen_cluster_key],
    normalize='index'
)

ax = composition.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Cell Type Composition: Anterior vs Posterior Hippocampus')
plt.xlabel('Region (group)')
plt.ylabel('Proportion of Cells')
plt.legend(title='DBSCAN Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


## 6. UMAP separated by Region
print("Generating comparative UMAPs...")
sc.pl.umap(adata_clean, color=[chosen_cluster_key, 'group'], wspace=0.4)


## 7. Gene Expression by Cluster AND Region (Using Auto-Discovered Markers)
print("Generating comparative dotplot...")

# Concatenate cluster label and region so each bar represents one cluster/region
# combination, e.g. "0_anterior", "1_posterior", etc.
adata_clean.obs['cluster_region'] = (
    adata_clean.obs[chosen_cluster_key].astype(str)
    + "_"
    + adata_clean.obs['group'].astype(str)
)

# Dynamically extract the top 5 marker genes per cluster from the DGE run above
dge_result = adata_clean.uns['rank_genes_groups']['names']
top_markers = []

for cluster_name in dge_result.dtype.names:
    top_markers.extend(dge_result[cluster_name][:5])

# Remove duplicates while preserving order
top_markers = list(dict.fromkeys(top_markers))

# Resolve gene IDs -> human-readable symbols if feature_name is available
if adata_clean.raw is not None and 'feature_name' in adata_clean.raw.var.columns:
    marker_symbols = adata_clean.raw.var.loc[top_markers, 'feature_name'].tolist()
else:
    marker_symbols = top_markers

sc.pl.dotplot(
    adata_clean,
    var_names=marker_symbols,
    groupby='cluster_region',
    gene_symbols='feature_name',
    standard_scale='var',
    title="Top Auto-Discovered Markers by Cluster and Region (DBSCAN)",
)


## 8. Global DGE: Anterior vs Posterior
print("Running DGE between anterior and posterior regions...")

sc.tl.rank_genes_groups(
    adata_clean,
    groupby='group',
    method="wilcoxon",
    use_raw=False,
)

sc.pl.rank_genes_groups_dotplot(
    adata_clean,
    n_genes=25,
    gene_symbols='feature_name',
    title="Top Regional Differences: Anterior vs Posterior (DBSCAN)",
)

