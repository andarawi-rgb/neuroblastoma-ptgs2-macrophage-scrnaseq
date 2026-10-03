import scanpy as sc
import pandas as pd
import numpy as np
import gc, resource

def mem():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

WORK = "/home/claude/nb_analysis/work"

adata_hvg = sc.read_h5ad(f"{WORK}/adata_hvg_only.h5ad")
print(f"Loaded HVG-only: {adata_hvg.shape}. peak GB: {mem():.2f}", flush=True)

sc.pp.scale(adata_hvg, max_value=10)
print(f"Scaled. peak GB: {mem():.2f}", flush=True)

sc.tl.pca(adata_hvg, n_comps=30, svd_solver="arpack")
print(f"PCA done. peak GB: {mem():.2f}", flush=True)

# drop the dense scaled matrix now that PCA is computed -- we only need obsm/obs going forward
adata_hvg.X = None
gc.collect()
print(f"Dropped dense X post-PCA. peak GB: {mem():.2f}", flush=True)

# Call harmonypy directly rather than scanpy's sc.external.pp.harmony_integrate wrapper:
# harmonypy 2.0's C++ backend returns Z_corr already as cells x dims, but scanpy's wrapper
# still does `.T` assuming the old pure-Python 0.0.x orientation (dims x cells), producing a
# degenerate shape. Same fix already used in the endometriosis pipeline (harmonypy 2.0
# shape-aware guard) -- apply it here too.
import harmonypy as hm
meta = adata_hvg.obs[["sample_id"]].copy()
ho = hm.run_harmony(adata_hvg.obsm["X_pca"], meta, ["sample_id"])
Z = np.asarray(ho.Z_corr)
if Z.shape[0] != adata_hvg.n_obs:
    Z = Z.T
assert Z.shape == (adata_hvg.n_obs, adata_hvg.obsm["X_pca"].shape[1]), f"unexpected Harmony output shape {Z.shape}"
adata_hvg.obsm["X_pca_harmony"] = Z
print(f"Harmony done (direct harmonypy call, shape-aware). peak GB: {mem():.2f}", flush=True)

np.save(f"{WORK}/harmony_pca.npy", adata_hvg.obsm["X_pca_harmony"])
np.save(f"{WORK}/pca_uncorrected.npy", adata_hvg.obsm["X_pca"])

sc.pp.neighbors(adata_hvg, use_rep="X_pca_harmony", n_neighbors=15)
print(f"Neighbors done. peak GB: {mem():.2f}", flush=True)

sc.tl.leiden(adata_hvg, resolution=1.0, key_added="leiden", flavor="igraph", n_iterations=2)
n_clusters = adata_hvg.obs["leiden"].nunique()
print(f"Leiden done. n_clusters={n_clusters}. peak GB: {mem():.2f}", flush=True)

adata_hvg.obs[["sample_id","leiden"]].to_csv(f"{WORK}/leiden_labels.csv")
print("Saved leiden_labels.csv, harmony_pca.npy, pca_uncorrected.npy. DONE 03b.", flush=True)
