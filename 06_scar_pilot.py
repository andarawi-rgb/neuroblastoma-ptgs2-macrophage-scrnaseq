import scanpy as sc
import numpy as np
import time, resource

def mem():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

WORK = "/home/claude/nb_analysis/work"
adata = sc.read_h5ad(f"{WORK}/adata_merged_all_cells.h5ad")  # RAW counts (pre-normalization)
print(f"Loaded raw counts: {adata.shape}. peak GB: {mem():.2f}", flush=True)

hvg_genes = sc.read_h5ad(f"{WORK}/adata_hvg_only.h5ad").var_names.tolist()
for g in ["PTGS2", "PTGES", "PTGS1"]:
    if g not in hvg_genes:
        hvg_genes.append(g)
adata_small = adata[:5000, [g for g in hvg_genes if g in adata.var_names]].copy()
print(f"Pilot subset: {adata_small.shape}. peak GB: {mem():.2f}", flush=True)

import scar
t0 = time.time()
m = scar.model(adata_small.X.toarray(), ambient_profile=None, batch_key=None, sparsity=0.9, verbose=True)
print(f"Model init took {time.time()-t0:.1f}s. peak GB: {mem():.2f}", flush=True)

t0 = time.time()
m.train(epochs=3, verbose=True)
dt = time.time() - t0
print(f"3 epochs took {dt:.1f}s -> {dt/3:.2f}s/epoch on 5000 cells x {adata_small.n_vars} genes. peak GB: {mem():.2f}", flush=True)
