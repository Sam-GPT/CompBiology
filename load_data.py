"""
load_data.py — load expression data into an AnnData object.

Two loaders:
  * load_tsv_data : original 4,994-cell TSV + label.ann (pre-normalized).
  * load_h5ad_data: full CellxGene h5ad (130k cells, includes raw counts).
"""

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc


def load_tsv_data(tsv_path: str = 'data.tsv',
                  ann_path: str = 'label.ann',
                  verbose: bool = True) -> ad.AnnData:
    """
    Read a TSV expression matrix and an annotation file into an AnnData.

    Returns
    -------
    AnnData with:
        adata.X                   — float32 expression matrix (cells x genes)
        adata.obs['cell_type']    — cell type labels from label.ann
        adata.obs['group']        — 'anterior' or 'posterior' (from barcode prefix)
        adata.obs['batch']        — donor+region prefix e.g. 'A67', 'P76'
        adata.var['feature_name'] — Ensembl gene IDs
    """

    # ------------------------------------------------------------------ #
    # 1. Read the expression matrix
    # ------------------------------------------------------------------ #
    print(f"[load_data] Reading {tsv_path} ...")
    df = pd.read_csv(tsv_path, sep='\t', index_col=0)

    if verbose:
        print(f"[load_data]   Raw shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"[load_data]   First row index:    {df.index[0]}")
        print(f"[load_data]   First column name:  {df.columns[0]}")

    # Auto-detect orientation by inspecting the index.
    # Ensembl gene IDs (ENSG...) or short all-caps gene symbols as rows = genes x cells -> transpose.
    first_idx = str(df.index[0])
    first_col = str(df.columns[0])
    idx_looks_like_gene = (
        first_idx.startswith('ENSG') or
        first_idx.startswith('ENSM') or
        (first_idx.isupper() and len(first_idx) <= 10)  # e.g. GAPDH, ACTB
    )
    col_looks_like_barcode = ('_' in first_col and len(first_col) > 10)

    if idx_looks_like_gene or col_looks_like_barcode:
        print("[load_data]   Detected genes x cells orientation -> transposing.")
        df = df.T   # now cells x genes
    else:
        print("[load_data]   Detected cells x genes orientation (no transpose).")

    if verbose:
        print(f"[load_data]   After orientation fix: {df.shape[0]} cells x {df.shape[1]} genes")

    # ------------------------------------------------------------------ #
    # 2. Read the annotation file
    # ------------------------------------------------------------------ #
    print(f"[load_data] Reading {ann_path} ...")

    # label.ann has NO header row, the first line is already real data.
    # We read with header=None and name the single data column 'cell_type'.
    ann = None
    for sep in ['\t', ',', ' ']:
        try:
            ann = pd.read_csv(ann_path, sep=sep, index_col=0, header=None,
                              names=['cell_type'])
            if ann.shape[1] >= 1:
                print(f"[load_data]   Annotation read (separator={repr(sep)}, no header).")
                break
        except Exception:
            pass

    if ann is None:
        raise RuntimeError(f"Could not read annotation file: {ann_path}")

    if verbose:
        print(f"[load_data]   Annotation shape: {ann.shape}")
        print(f"[load_data]   First few rows:\n{ann.head(4)}")

    # ------------------------------------------------------------------ #
    # 3. Align cell barcodes between matrix and annotation
    # ------------------------------------------------------------------ #
    common = df.index.intersection(ann.index)
    if len(common) == len(df):
        print("[load_data]   Cell barcodes matched perfectly.")
        ann = ann.loc[df.index]
    elif len(common) > 0:
        print(f"[load_data]   {len(common)}/{len(df)} barcodes matched - "
              f"dropping {len(df) - len(common)} unmatched cell(s).")
        df  = df.loc[common]
        ann = ann.loc[common]
    else:
        print("[load_data]   No index overlap - aligning by row position.")
        min_len = min(len(df), len(ann))
        df  = df.iloc[:min_len]
        ann = ann.iloc[:min_len]
        ann.index = df.index

    # ------------------------------------------------------------------ #
    # 4. Build AnnData
    # ------------------------------------------------------------------ #
    X = df.values.astype(np.float32)
    adata = ad.AnnData(X=X, obs=ann.copy(), var=pd.DataFrame(index=df.columns))

    # pipeline.py uses adata.var["feature_name"] for gene name lookups
    adata.var['feature_name'] = adata.var.index.astype(str)

    # ------------------------------------------------------------------ #
    # 5. Derive 'group' and 'batch' from cell barcodes
    # ------------------------------------------------------------------ #
    # Barcodes follow the pattern {region}{patient}_{sequence}
    # e.g.  A67_TACACGACACGGTGTC  ->  Anterior hippocampus, patient 67
    #       P67_TTCTCTCGTTCTCAGA  ->  Posterior hippocampus, patient 67
    barcodes = adata.obs.index.astype(str)
    prefix   = barcodes.str.split('_').str[0]   # 'A67', 'P76', etc.

    adata.obs['group'] = prefix.str[0].map({'A': 'anterior', 'P': 'posterior'})
    adata.obs['batch'] = prefix

    unmapped = adata.obs['group'].isna().sum()
    if unmapped > 0:
        print(f"[load_data]   WARNING: {unmapped} cells had unrecognised barcode prefix. "
              "Expected 'A' (anterior) or 'P' (posterior) as first character.")
    else:
        print(f"[load_data]   'group': {dict(adata.obs['group'].value_counts())}")
        print(f"[load_data]   'batch': {sorted(adata.obs['batch'].unique())}")

    # ------------------------------------------------------------------ #
    # 6. Summary
    # ------------------------------------------------------------------ #
    print(f"\n[load_data] AnnData ready: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"            obs columns : {list(adata.obs.columns)}")

    return adata


def load_h5ad_data(path: str,
                   n_cells: int = 10000,
                   random_state: int = 42,
                   verbose: bool = True) -> ad.AnnData:
    """
    Load a CellxGene h5ad and randomly subsample.

    The CellxGene h5ad stores log-normalized values in .X and raw integer counts
    in .raw.X. We swap .X to the raw counts so the downstream pipeline (which
    expects raw counts and runs normalize_total + log1p itself) works correctly.

    Required obs columns in the source h5ad: 'group' (anterior/posterior),
    'batch', 'cell_type'. Required var column: 'feature_name'.
    """
    print(f"[load_data] Reading {path} ...")
    adata = sc.read_h5ad(path)
    if verbose:
        print(f"[load_data]   Full shape: {adata.n_obs} cells x {adata.n_vars} genes")

    if adata.raw is None:
        raise RuntimeError(
            f"{path} has no .raw — cannot recover raw counts. "
            "Pipeline requires raw counts for HVG selection (seurat_v3)."
        )
    adata.X = adata.raw.X.copy()
    adata.raw = None
    if verbose:
        print("[load_data]   Replaced .X with raw integer counts from .raw.X")

    if n_cells is not None and n_cells < adata.n_obs:
        sc.pp.subsample(adata, n_obs=n_cells, random_state=random_state)
        if verbose:
            print(f"[load_data]   Subsampled to {adata.n_obs} cells (seed={random_state}).")

    for col in ('group', 'batch', 'cell_type'):
        if col not in adata.obs.columns:
            raise RuntimeError(f"Expected obs column '{col}' not found in {path}.")
    if 'feature_name' not in adata.var.columns:
        raise RuntimeError(f"Expected var column 'feature_name' not found in {path}.")

    if verbose:
        print(f"[load_data]   'group': {dict(adata.obs['group'].value_counts())}")
        print(f"[load_data]   'batch': {sorted(adata.obs['batch'].astype(str).unique())}")
        print(f"\n[load_data] AnnData ready: {adata.n_obs} cells x {adata.n_vars} genes")
        print(f"            obs columns : {list(adata.obs.columns)[:10]}...")

    return adata


# Run this file directly to verify if data loads correctly before running pipeline.py
if __name__ == '__main__':
    adata = load_tsv_data('data.tsv', 'label.ann')
    print("\nadata.obs.head():")
    print(adata.obs.head())
    print("\nadata.var.head():")
    print(adata.var.head())
