import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import scipy.sparse as sp
import gc, resource

def mem():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

WORK = "/home/claude/nb_analysis/work"

adata = sc.read_h5ad(f"{WORK}/adata_merged_all_cells.h5ad")
print(f"Loaded: {adata.shape}. peak GB: {mem():.2f}", flush=True)

adata.layers["counts"] = adata.X.copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(f"Normalized+log1p. peak GB: {mem():.2f}", flush=True)

sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="sample_id")
print(f"HVG selected: {adata.var['highly_variable'].sum()}. peak GB: {mem():.2f}", flush=True)

adata.raw = adata  # keep full log-normalized matrix for target-gene quantification later

adata_hvg = adata[:, adata.var["highly_variable"]].copy()
sc.pp.scale(adata_hvg, max_value=10)
print(f"Scaled HVG subset: {adata_hvg.shape}. peak GB: {mem():.2f}", flush=True)

sc.tl.pca(adata_hvg, n_comps=30, svd_solver="arpack")
print(f"PCA done. peak GB: {mem():.2f}", flush=True)

sc.external.pp.harmony_integrate(adata_hvg, key="sample_id")
print(f"Harmony done. peak GB: {mem():.2f}", flush=True)

np.save(f"{WORK}/harmony_pca.npy", adata_hvg.obsm["X_pca_harmony"])

sc.pp.neighbors(adata_hvg, use_rep="X_pca_harmony", n_neighbors=15)
print(f"Neighbors done. peak GB: {mem():.2f}", flush=True)

sc.tl.leiden(adata_hvg, resolution=1.0, key_added="leiden", flavor="igraph", n_iterations=2)
print(f"Leiden done. n_clusters={adata_hvg.obs['leiden'].nunique()}. peak GB: {mem():.2f}", flush=True)

adata_hvg.obs[["sample_id","leiden"]].to_csv(f"{WORK}/leiden_labels.csv")
print("Saved leiden_labels.csv and harmony_pca.npy. DONE clustering step.", flush=True)
