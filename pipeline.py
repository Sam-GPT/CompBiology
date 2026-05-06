import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import scanpy.external as sce
import harmonypy as hm



sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=150, facecolor='white')


# Read in the data
adata = sc.read_h5ad('data.h5ad')

# Limit to 30K cells
adata = adata[:30000, :].copy()

## 1. Normalize / Preprocess

# Annotate gene populations
# mitochondrial genes, "MT-" for human, "Mt-" for mouse
adata.var["mt"] = adata.var_names.str.startswith("MT-")
# ribosomal genes
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
# hemoglobin genes
adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

# Calculate QC metrics
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)


# Visualize QC metrics
# n_genes_by_counts: Number of genes detected in a cell
# total_counts: Total number of molecules (UMIs) in a cell
# pct_counts_mt: Percentage of mitochondrial genes in a cell
# sc.pl.violin(
#     adata,
#     ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
#     jitter=0.4,
#     multi_panel=True,
# )

# Filter cells
sc.pp.filter_cells(adata, min_genes=100) # Keeps only cells that express at least 100 genes
sc.pp.filter_genes(adata, min_cells=3) # Keeps only genes that are expressed in at least 3 cells


# Doublet Detection
sc.pp.scrublet(adata, batch_key="batch") # doublet: which are multiple cells captured in one droplet.
#sc.pl.umap(adata, color=["doublet_score", "predicted_doublet"])

adata = adata[adata.obs["doublet_score"] < 0.25].copy()

# Normalization and Feature Selection

# Saving count data
adata.layers["counts"] = adata.X.copy()

# Normalizing to median total counts
sc.pp.normalize_total(adata)
# Logarithmize the data
sc.pp.log1p(adata)

# Identify 2,000 highly variable genes (HVGs)
sc.pp.highly_variable_genes(
    adata, 
    n_top_genes=2000, 
    batch_key="batch",
    subset=False,
    flavor="seurat_v3",
)

adata = adata[:, adata.var["highly_variable"]].copy()



## 2. Batch Effect Correction

# Dimensionality reduction using PCA
sc.tl.pca(adata, svd_solver="arpack")



#sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True)



Z = adata.obsm["X_pca"]   # (9769, 50)

# run harmony directly
ho = hm.run_harmony(
    Z,
    adata.obs,
    vars_use=["batch"]
)

# correct orientation fix
adata.obsm["X_pca_harmony"] = ho.Z_corr.T


# # Before Harmony Batch correction
# sc.pl.pca(adata, color="batch")

# # After Harmony Batch correction
# sc.pl.embedding(
#     adata,
#     basis="X_pca_harmony",
#     color="batch"
# )

# Constructing the neighborhood graph using the Harmony-corrected PCA embeddings
sc.pp.neighbors(adata, use_rep="X_pca_harmony")
sc.tl.umap(adata)


sc.pl.umap(
    adata,
    color="batch",
    # Setting a smaller point size to get prevent overlap
    size=2,
)

## 3. Clustering (Benchmarking Leiden Resolutions)
print("Running clustering...")

# Method: Leiden algorithm (The modern standard)
# We test multiple resolutions to benchmark how it affects the number of clusters (as seen on slide 35)

sc.tl.leiden(adata, resolution=0.25, key_added="leiden_res_0.25")
sc.tl.leiden(adata, resolution=0.5, key_added="leiden_res_0.50")
sc.tl.leiden(adata, resolution=1.0, key_added="leiden_res_1.00")

# Visualize the clustering results side-by-side on the UMAP for your benchmark report
sc.pl.umap(
    adata,
    color=["leiden_res_0.25", "leiden_res_0.50", "leiden_res_1.00"],
    wspace=0.4,
    title=["Leiden (Res=0.25)", "Leiden (Res=0.50)", "Leiden (Res=1.0)"]
)

## 4. Cluster Interpretation (Finding meaning in the presence of noise)
print("Running Differential Gene Expression to find marker genes...")

# Let's proceed with the Leiden algorithm at 0.50 resolution for our interpretation
chosen_cluster_key = "leiden_res_0.50"

# Rank genes to find cluster-specific marker genes
# 'wilcoxon' is the standard non-parametric statistical test used for this
sc.tl.rank_genes_groups(
    adata,
    groupby=chosen_cluster_key,
    method="wilcoxon",
    use_raw=False
)

# 4a. Visualize the top 5 marker genes for each cluster using a Dotplot
# Dotplots are excellent for interpreting clusters (as shown on slide 36)
# It shows both the mean expression (color) and fraction of cells expressing the gene (dot size)
sc.pl.rank_genes_groups_dotplot(
    adata,
    n_genes=5,
    groupby=chosen_cluster_key,
    standard_scale="var", # Scales expression between 0 and 1 for easier visual comparison
    title="Top 5 Marker Genes per Cluster"
)

# 4b. Extract the marker genes into a DataFrame to investigate biologically
# Let's say you want to look at the top markers for Cluster '0'
cluster_0_markers = sc.get.rank_genes_groups_df(adata, group="0")

print("\n--- Top 10 marker genes for Cluster 0 ---")
print(cluster_0_markers.head(10))

# Note for your assignment report:
# Once you have these gene lists, you would typically look them up in biological databases
# (like CellMarker or literature) to say "Cluster 0 is highly expressing CD14, so it is a Monocyte."