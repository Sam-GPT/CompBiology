import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from pipeline import adata

CLUSTER_COL = 'sc3_k6'

# 1. Cell Type Composition by Region
print("Generating composition plot...")
composition = pd.crosstab(
    adata.obs['group'], adata.obs[CLUSTER_COL], normalize='index'
)
ax = composition.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Cell Type Composition: Anterior vs Posterior Hippocampus')
plt.xlabel('Region (group)')
plt.ylabel('Proportion of Cells')
plt.legend(title=f'SC3 Cluster (k={CLUSTER_COL.split("k")[1]})',
           bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('figures/composition_anterior_vs_posterior.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. UMAP separated by Region
print("Generating comparative UMAPs...")
sc.pl.umap(adata, color=[CLUSTER_COL, 'group'], wspace=0.4, save='_sc3_vs_region.png')

# 3. Gene Expression by Cluster AND Region
print("Generating comparative dotplot...")
adata.obs['cluster_region'] = (
        adata.obs[CLUSTER_COL].astype(str) + '_' + adata.obs['group'].astype(str)
)

# Dynamically extract top 5 marker genes per cluster from the DGE run above
dge_result = adata.uns['rank_genes_groups']['names']
top_markers = []
for cluster_name in dge_result.dtype.names:
    top_markers.extend(dge_result[cluster_name][:5])
top_markers = list(dict.fromkeys(top_markers))  # deduplicate, preserve order

if 'feature_name' in adata.var.columns:
    marker_symbols = adata.var.loc[top_markers, 'feature_name'].tolist()
else:
    marker_symbols = top_markers

sc.pl.dotplot(
    adata,
    var_names=marker_symbols,
    groupby='cluster_region',
    gene_symbols='feature_name' if 'feature_name' in adata.var.columns else None,
    standard_scale='var',
    title=f'Top Auto-Discovered Markers by Cluster and Region ({CLUSTER_COL})',
    save='_marker_by_cluster_region.png',
)

# 4. Global DGE: Anterior vs Posterior
print("Running DGE between anterior and posterior regions...")
sc.tl.rank_genes_groups(adata, groupby='group', method='wilcoxon')

sc.pl.rank_genes_groups_dotplot(
    adata,
    n_genes=25,
    gene_symbols='feature_name' if 'feature_name' in adata.var.columns else None,
    title='Top Regional Differences: Anterior vs Posterior',
    save='_anterior_vs_posterior_dge.png',
)
