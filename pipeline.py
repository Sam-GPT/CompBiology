import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import scanpy.external as sce
import harmonypy as hm
import os
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score



sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=150, facecolor='white')


# Read in the data
adata = sc.read_h5ad('data.h5ad')

# Limit to 3K cells
adata = adata[:3000, :].copy()

print("Variable columns:", adata.var.columns)
print("Observation columns:", adata.obs.columns)



## 1. Normalize / Preprocess

# Annotate gene populations
# mitochondrial genes, "MT-" for human, "Mt-" for mouse
adata.var["mt"] = adata.var["feature_name"].str.upper().str.startswith("MT-")
# ribosomal genes
adata.var["ribo"] = adata.var["feature_name"].str.upper().str.startswith(("RPS", "RPL"))
# hemoglobin genes
adata.var["hb"] = adata.var["feature_name"].str.upper().str.contains("^HB[^(P)]")



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
sc.pp.filter_cells(adata, min_genes=200) # Keeps only cells that express at least 200 genes
sc.pp.filter_genes(adata, min_cells=3) # Keeps only genes that are expressed in at least 3 cells


# Doublet Detection
sc.pp.scrublet(adata, batch_key="batch") # doublet: which are multiple cells captured in one droplet.

# Visualize doublet scores and predicted doublets 
# sc.pl.umap(adata, color=["doublet_score", "predicted_doublet"])

adata = adata[adata.obs["doublet_score"] < 0.56].copy()

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
    layer="counts",
)

# ADD THIS LINE: Save the normalized/log-transformed data including all genes
adata.raw = adata

# Subsetting to highly variable genes
adata = adata[:, adata.var["highly_variable"]].copy()



## 2. Batch Effect Correction

# Dimensionality reduction using PCA
sc.tl.pca(adata, svd_solver="arpack")


# Visualizes how much variance the first 50 PCA dimensions explain (on a log scale)
# sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True)



Z = adata.obsm["X_pca"]   # (9769, 50)

# run harmony directly
ho = hm.run_harmony(
    Z,
    adata.obs,
    vars_use=["batch"]
)

# correct orientation fix
adata.obsm["X_pca_harmony"] = ho.Z_corr.T


# # Before Harmony Batch correction (Plotting the PCA colored by batch to see the batch effect)
# sc.pl.pca(adata, color="batch")

# # After Harmony Batch correction (Plotting the Harmony-corrected PCA colored by batch to see if the batch effect is reduced)
# sc.pl.embedding(
#     adata,
#     basis="X_pca_harmony",
#     color="batch"
# )

# Constructing the neighborhood graph using the Harmony-corrected PCA embeddings
sc.pp.neighbors(adata, use_rep="X_pca_harmony")
sc.tl.umap(adata)

# Visualize the UMAP colored by batch to check if the batch effect has been mitigated
# sc.pl.umap(
#     adata,
#     color="batch",
#     # Setting a smaller point size to get prevent overlap
#     size=2,
# )





# adata.to_df().T.to_csv("data/MyDataset/data.tsv", sep="\t")

# adata.obs["cell_type"].to_csv(
#     "data/MyDataset/label.ann",
#     sep="\t",
#     header=False
# )



print("Done")


## 3. Clustering
print("Running clustering...")

# Method: Leiden algorithm (The modern standard)
# We test multiple resolutions to benchmark how it affects the number of clusters (as seen on slide 35)

# sc.tl.leiden(adata, resolution=0.25, key_added="leiden_res_0.25")
# sc.tl.leiden(adata, resolution=0.5, key_added="leiden_res_0.50")
# sc.tl.leiden(adata, resolution=1.0, key_added="leiden_res_1.00")

# # Visualize the clustering results side-by-side on the UMAP for your benchmark report
# sc.pl.umap(
#     adata,
#     color=["leiden_res_0.25", "leiden_res_0.50", "leiden_res_1.00"],
#     wspace=0.4,
#     title=["Leiden (Res=0.25)", "Leiden (Res=0.50)", "Leiden (Res=1.0)"]
# )





predicted_labels = pd.read_csv('result/pred_MyDataset.txt', sep='\t')

adata.obs['scDFC_cluster'] = predicted_labels['label'].astype(str).values



# Markers we saw in your earlier successful plot:
confirmed_markers = ['CTNNA3', 'PITPNC1', 'TNR', 'RBFOX3', 'PLXNA4', 'LRMDA']

# Filter just in case
to_plot = [g for g in confirmed_markers if g in adata.var['feature_name'].values]

# sc.pl.umap(adata, color=to_plot, gene_symbols='feature_name', ncols=3)











## 4. Cluster Interpretation (Finding meaning in the presence of noise)
print("Running Differential Gene Expression to find marker genes...")


sc.tl.rank_genes_groups(
    adata,
    groupby="scDFC_cluster",
    method="wilcoxon"
)


# sc.pl.rank_genes_groups(adata)
# Use 'feature_name' to tell Scanpy where the readable symbols are
# sc.pl.rank_genes_groups(adata, n_genes=20, gene_symbols='feature_name', sharey=False)

# 1. Cell Type Composition by Region
print("Generating composition plot...")
composition = pd.crosstab(adata.obs['group'], adata.obs['scDFC_cluster'], normalize='index')

ax = composition.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Cell Type Composition: Anterior vs Posterior Hippocampus')
plt.xlabel('Region (group)')
plt.ylabel('Proportion of Cells')
plt.legend(title='scDFC Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()



os.makedirs("figures", exist_ok=True)
plt.savefig("figures/composition_bar.png", dpi=300, bbox_inches='tight')
plt.close() # Closes the plot so it doesn't wait for your input

# 2. UMAP separated by Region
print("Generating comparative UMAPs...")
sc.pl.umap(adata, color=['scDFC_cluster', 'group'], wspace=0.4, save="_comparative.png", show=False)

# 3. Gene Expression by Cluster AND Region (Using Auto-Discovered Markers)
print("Generating comparative dotplot...")

adata.obs['cluster_region'] = adata.obs['scDFC_cluster'].astype(str) + "_" + adata.obs['group'].astype(str)


# Dynamically extract the top 5 marker genes for each cluster from the previous DGE run
dge_result = adata.uns['rank_genes_groups']['names']
top_markers = []

for cluster_name in dge_result.dtype.names:
    top_markers.extend(dge_result[cluster_name][:5]) 

# Remove any duplicates 
top_markers = list(dict.fromkeys(top_markers))


if 'feature_name' in adata.raw.var.columns:
    marker_symbols = adata.raw.var.loc[top_markers, 'feature_name'].tolist()
else:
    marker_symbols = top_markers

sc.pl.dotplot(
    adata, 
    var_names=marker_symbols, 
    groupby='cluster_region', 
    gene_symbols='feature_name',
    standard_scale='var',
    title="Top Auto-Discovered Markers by Cluster and Region",
    save="_cluster_region_markers.png",
    show=False
)

# 4. Global DGE: Anterior vs Posterior
print("Running DGE between anterior and posterior regions...")
sc.tl.rank_genes_groups(
    adata,
    groupby='group', 
    method="wilcoxon"
)

# Visualize the top 25 genes driving the difference between regions
sc.pl.rank_genes_groups_dotplot(
    adata, 
    n_genes=25, 
    gene_symbols='feature_name', 
    title="Top Regional Differences: Anterior vs Posterior",
    save="_anterior_vs_posterior.png",
    show=False
)

# Clustering Benchmarks

embed = adata.obsm["X_pca_harmony"]

labels = adata.obs["scDFC_cluster"].astype("category").cat.codes.values

sil = silhouette_score(embed, labels, sample_size=min(5000, len(labels)))

db = davies_bouldin_score(embed, labels)

true_labels = adata.obs["cell_type"].astype(str).values
ari = adjusted_rand_score(true_labels, adata.obs["scDFC_cluster"].astype(str).values)
print(f"Metrics -> Silhouette: {sil:.4f}, Davies-Bouldin: {db:.4f}, ARI: {ari:.4f}")





