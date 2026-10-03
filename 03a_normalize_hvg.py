import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import gc, resource

def mem():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

WORK = "/home/claude/nb_analysis/work"

adata = sc.read_h5ad(f"{WORK}/adata_merged_all_cells.h5ad")
print(f"Loaded: {adata.shape}. peak GB: {mem():.2f}", flush=True)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(f"Normalized+log1p. peak GB: {mem():.2f}", flush=True)

# Save the full log-normalized matrix to disk NOW, then drop it from this process's
# working set for the HVG subset (memory-safe split, matches the successful
# pattern used in the endometriosis pipeline: separate processes for norm vs PCA/Harmony).
adata.write(f"{WORK}/adata_norm_full.h5ad")
print(f"Saved adata_norm_full.h5ad (full {adata.n_vars} genes). peak GB: {mem():.2f}", flush=True)

sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="sample_id")
n_hvg = int(adata.var["highly_variable"].sum())
print(f"HVG selected: {n_hvg}. peak GB: {mem():.2f}", flush=True)

adata_hvg = adata[:, adata.var["highly_variable"]].copy()
del adata
gc.collect()
adata_hvg.write(f"{WORK}/adata_hvg_only.h5ad")
print(f"Saved adata_hvg_only.h5ad: {adata_hvg.shape}. peak GB: {mem():.2f}", flush=True)
print("DONE 03a.", flush=True)
