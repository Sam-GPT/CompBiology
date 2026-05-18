import os

# Note: GiniClust3 was prototyped but excluded from the unified benchmark. See git history.
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import scanpy.external as sce
import harmonypy as hm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

sc.settings.verbosity = 3
sc.settings.autoshow = False
sc.settings.set_figure_params(dpi=150, facecolor='white')

os.makedirs("output", exist_ok=True)
sc.settings.figdir = "output"


# Read in the data
adata = sc.read_h5ad('data.h5ad')

# CellxGene stores log-normalized values in .X and raw integer counts in .raw.X.
# Swap so the pipeline's normalize_total + log1p + seurat_v3 HVG step works on raw counts.
adata.X = adata.raw.X.copy()
adata.raw = None

# Limit to 30K cells
adata = adata[:30000, :].copy()


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












## 3. Clustering
print("Running clustering...")

# Method: Leiden algorithm (The modern standard)
# We test multiple resolutions to benchmark how it affects the number of clusters (as seen on slide 35)

sc.tl.leiden(adata, resolution=0.25, key_added="leiden_res_0.25", flavor="igraph", directed=False, n_iterations=2)
sc.tl.leiden(adata, resolution=0.5,  key_added="leiden_res_0.50", flavor="igraph", directed=False, n_iterations=2)
sc.tl.leiden(adata, resolution=1.0,  key_added="leiden_res_1.00", flavor="igraph", directed=False, n_iterations=2)

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
sc.tl.dendrogram(adata, groupby=chosen_cluster_key, use_rep="X_pca_harmony")

# 4a. Visualize the top 5 marker genes for each cluster using a Dotplot
# Dotplots are excellent for interpreting clusters (as shown on slide 36)
# It shows both the mean expression (color) and fraction of cells expressing the gene (dot size)
sc.pl.rank_genes_groups_dotplot(
    adata,
    n_genes=5,
    groupby=chosen_cluster_key,
    standard_scale="var", # Scales expression between 0 and 1 for easier visual comparison
    title="Top 5 Marker Genes per Cluster",
    save="leiden_markers.png",
)

# 4b. Extract the marker genes into a DataFrame to investigate biologically
# Let's say you want to look at the top markers for Cluster '0'
cluster_0_markers = sc.get.rank_genes_groups_df(adata, group="0")

print("\n--- Top 10 marker genes for Cluster 0 ---")
print(cluster_0_markers.head(10))

# Note for your assignment report:
# Once you have these gene lists, you would typically look them up in biological databases
# (like CellMarker or literature) to say "Cluster 0 is highly expressing CD14, so it is a Monocyte."




## 5. Benchmarking Alternative Algorithms
import scipy.sparse

# ================================================================================
# ALGORITHM 1: K-Means (Distance-based clustering)
# ================================================================================
print("\nRunning K-Means for algorithm benchmark...")

# We force K-Means to find 9 clusters (since Leiden res=0.50 found 9 clusters)
kmeans = KMeans(n_clusters=9, random_state=42)
kmeans.fit(adata.obsm['X_pca_harmony'])

# Save the K-Means results in the adata object
adata.obs['kmeans'] = kmeans.labels_.astype(str)
adata.obs['kmeans'] = adata.obs['kmeans'].astype('category')


# ================================================================================
# FINAL BENCHMARK PLOT
# ================================================================================

# Plot Leiden and K-Means side by side on the UMAP
sc.pl.umap(
    adata,
    color=["leiden_res_0.50", "kmeans"],
    wspace=0.4,
    title=["Leiden (Graph)", "K-Means (Distance)"]
)

## 6. Differential Gene Expression on K-Means clusters
# Wilcoxon DGE so K-Means gets its own marker-gene interpretation
# (Section 4 above already produces this for Leiden).

print("\nRunning DGE on K-Means clusters...")
sc.tl.rank_genes_groups(adata, groupby="kmeans", method="wilcoxon", use_raw=False)
sc.tl.dendrogram(adata, groupby="kmeans", use_rep="X_pca_harmony")
sc.pl.rank_genes_groups_dotplot(
    adata,
    n_genes=5,
    groupby="kmeans",
    standard_scale="var",
    title="Top 5 Marker Genes per K-Means Cluster",
    save="kmeans_markers.png",
)


## 7. Cell Type Composition: Anterior vs Posterior Hippocampus
# Stacked bar of K-Means cluster proportions per region.

for alg_col, alg_name, fname in [
    ("kmeans", "K-Means", "composition_kmeans_anterior_vs_posterior.png"),
]:
    composition = pd.crosstab(
        adata.obs["group"], adata.obs[alg_col], normalize="index"
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    composition.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(f"Cell Type Composition: Anterior vs Posterior Hippocampus ({alg_name})")
    ax.set_xlabel("Region")
    ax.set_ylabel("Proportion of Cells")
    ax.legend(title=f"{alg_name} Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"output/{fname}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved output/{fname}")


## 8. Regional DGE: Anterior vs Posterior
# Globally compare anterior vs posterior cells, ignoring clustering.

print("\nRunning regional DGE (anterior vs posterior)...")
sc.tl.rank_genes_groups(adata, groupby="group", method="wilcoxon", use_raw=False)
sc.pl.rank_genes_groups_dotplot(
    adata,
    n_genes=25,
    groupby="group",
    title="Top Regional Differences: Anterior vs Posterior",
    save="anterior_vs_posterior_dge.png",
)


## 9. Quantitative Benchmark Metrics
# Silhouette + Davies-Bouldin + ARI vs Leiden + ARI vs ground truth on Harmony-corrected PCA.

print("\nComputing benchmark metrics (silhouette, Davies-Bouldin, ARI vs Leiden, ARI vs truth)...")
embed = adata.obsm["X_pca_harmony"]
leiden_ref = adata.obs["leiden_res_0.50"].astype(str).astype("category").cat.codes.values
truth_ref = adata.obs["cell_type"].astype(str).astype("category").cat.codes.values

bench_rows = []
for name, col in [
    ("Leiden (res=0.50)", "leiden_res_0.50"),
    ("K-Means", "kmeans"),
]:
    labels = adata.obs[col].astype(str).astype("category").cat.codes.values
    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        print(f"  Skipping {name}: only {n_clusters} cluster(s)")
        continue
    sample_size = min(5000, len(labels))
    sil = silhouette_score(embed, labels, sample_size=sample_size)
    db = davies_bouldin_score(embed, labels)
    ari = adjusted_rand_score(leiden_ref, labels)
    ari_truth = adjusted_rand_score(truth_ref, labels)
    bench_rows.append({
        "method": name,
        "n_clusters": n_clusters,
        "silhouette": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "ari_vs_leiden": round(ari, 4),
        "ari_vs_truth": round(ari_truth, 4),
    })

bench_df = pd.DataFrame(bench_rows)
print("\n=== Benchmark Metrics ===")
print(bench_df.to_string(index=False))

# Grouped bar chart: one panel per metric, best value highlighted.
fig, axes = plt.subplots(1, 4, figsize=(19, 4))
metric_specs = [
    ("silhouette", "Silhouette score\n(higher = better)", "max"),
    ("davies_bouldin", "Davies-Bouldin index\n(lower = better)", "min"),
    ("ari_vs_leiden", "ARI vs Leiden\n(higher = better)", "max"),
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
fig.suptitle("K-Means vs Leiden — clustering benchmark", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("output/benchmark_kmeans.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved output/benchmark_kmeans.png")

print("\nBenchmark complete!")