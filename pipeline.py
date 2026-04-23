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