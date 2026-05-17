import logging

import scanpy as sc
import harmonypy as hm
from load_data import load_h5ad_data
from sc3 import sc3_cluster, sc3_benchmark_plot

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
# Full CellxGene h5ad has 130k cells × 17k genes; we subsample to 10k for runtime.
# feature_name (gene symbols) and group/batch obs columns are already present.
adata = load_h5ad_data('703771a1-236f-4eda-9c04-318d882e149b.h5ad', n_cells=10000)


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

# Filter cells
# Full-genome data (~17k genes); 200 is the standard 10x Genomics QC floor.
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Doublet Detection
# Now that we load raw counts from the h5ad's .raw layer, Scrublet works correctly.
sc.pp.scrublet(adata, batch_key="batch")
adata = adata[~adata.obs["predicted_doublet"]].copy()

# Normalization and Feature Selection

# Saving count data
adata.layers["counts"] = adata.X.copy()

# Normalizing to median total counts
sc.pp.normalize_total(adata)
# Logarithmize the data
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

# Visualizes how much variance the first 50 PCA dimensions explain
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True, save="_pca_variance.png")

Z = adata.obsm["X_pca"]

# Run Harmony batch correction
ho = hm.run_harmony(
    Z,
    adata.obs,
    vars_use=["batch"]
)
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

# Constructing the neighborhood graph using the Harmony-corrected PCA embeddings
sc.pp.neighbors(adata, use_rep="X_pca_harmony")
sc.tl.umap(adata)

# UMAP coloured by batch, check batch effect is reduced
sc.pl.umap(
    adata,
    color="batch",
    size=2,
    title="UMAP coloured by batch (after Harmony)",
    save="_umap_batch.png",
)

# UMAP coloured by cell type, sanity check using ground-truth labels
sc.pl.umap(
    adata,
    color="cell_type",
    title="UMAP coloured by cell type",
    save="_umap_cell_type.png",
)


## 3. Clustering
print("Running clustering...")

sc.tl.leiden(adata, resolution=0.25, key_added="leiden_res_0.25", flavor="igraph", directed=False, n_iterations=2)
sc.tl.leiden(adata, resolution=0.5,  key_added="leiden_res_0.50", flavor="igraph", directed=False, n_iterations=2)
sc.tl.leiden(adata, resolution=1.0,  key_added="leiden_res_1.00", flavor="igraph", directed=False, n_iterations=2)

# Visualize the clustering results side-by-side on the UMAP
sc.pl.umap(
    adata,
    color=["leiden_res_0.25", "leiden_res_0.50", "leiden_res_1.00"],
    wspace=0.4,
    title=["Leiden (Res=0.25)", "Leiden (Res=0.50)", "Leiden (Res=1.0)"],
    save="_leiden_comparison.png",
)


sc3_cluster(adata, k_range=range(4, 8), n_subsample=2000, basis='X_pca_harmony', random_state=42)

sc.pl.umap(
    adata,
    color=['sc3_k5', 'sc3_k6', 'leiden_res_0.50'],
    wspace=0.4,
    title=['SC3 (k=5)', 'SC3 (k=6)', 'Leiden (Res=0.50)'],
    save='_sc3_vs_leiden.png',
)

sc3_benchmark_plot(adata, k_range=range(4, 8), leiden_key='leiden_res_0.50',
                   basis='X_pca_harmony', save='figures/sc3_benchmark.png')


## 4. Cluster Interpretation
print("Running Differential Gene Expression to find marker genes...")

chosen_cluster_key = "leiden_res_0.50"

sc.tl.rank_genes_groups(
    adata,
    groupby=chosen_cluster_key,
    method="wilcoxon",
    use_raw=False,
)

sc.tl.dendrogram(adata, groupby=chosen_cluster_key, use_rep="X_pca_harmony")

# Dotplot: top 5 marker genes per cluster, using readable gene symbols
sc.pl.rank_genes_groups_dotplot(
    adata,
    n_genes=5,
    groupby=chosen_cluster_key,
    gene_symbols="feature_name",
    standard_scale="var",
    title="Top 5 Marker Genes per Cluster",
    save="_marker_genes.png",
)

# Print top 10 markers for cluster 0 with readable names
cluster_0_markers = sc.get.rank_genes_groups_df(adata, group="0")
cluster_0_markers["gene_symbol"] = (
    adata.var.loc[cluster_0_markers["names"], "feature_name"].values
)

print("\n--- Top 10 marker genes for Cluster 0 ---")
print(cluster_0_markers[["gene_symbol", "names", "scores", "logfoldchanges", "pvals_adj"]].head(10))
