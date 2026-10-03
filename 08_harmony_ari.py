import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score
import gc, resource

def mem():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

WORK = "/home/claude/nb_analysis/work"

harmony_pca = np.load(f"{WORK}/harmony_pca.npy")
uncorrected_pca = np.load(f"{WORK}/pca_uncorrected.npy")
leiden_harmony = pd.read_csv(f"{WORK}/leiden_labels.csv", index_col=0)

a = ad.AnnData(X=np.zeros((uncorrected_pca.shape[0], 1), dtype=np.float32))
a.obs_names = leiden_harmony.index
a.obsm["X_pca"] = uncorrected_pca
print(f"Built minimal AnnData for uncorrected-PCA clustering: {a.shape}. peak GB: {mem():.2f}", flush=True)

sc.pp.neighbors(a, use_rep="X_pca", n_neighbors=15)
sc.tl.leiden(a, resolution=1.0, key_added="leiden_uncorrected", flavor="igraph", n_iterations=2)
n_clusters_uncorrected = a.obs["leiden_uncorrected"].nunique()
print(f"Uncorrected (PCA-only) Leiden clustering: {n_clusters_uncorrected} clusters. peak GB: {mem():.2f}", flush=True)

ari = adjusted_rand_score(leiden_harmony["leiden"].astype(str), a.obs["leiden_uncorrected"].astype(str))
print(f"\nAdjusted Rand Index (Harmony-corrected vs uncorrected Leiden clustering): ARI = {ari:.4f}", flush=True)

a.obs[["leiden_uncorrected"]].to_csv(f"{WORK}/leiden_uncorrected_labels.csv")

# also: cell-type-level concordance -- map uncorrected clusters to cell types via the SAME marker approach
# quick proxy: for each harmony cell_type, what fraction of its cells fall into a single dominant uncorrected cluster
obs_ann = pd.read_csv(f"{WORK}/obs_annotated.csv", index_col=0)
obs_ann = obs_ann.loc[a.obs_names]
ct = obs_ann["cell_type"]
uncorr = a.obs["leiden_uncorrected"]
print("\n=== Per Harmony-cell-type: dominant uncorrected-cluster purity ===", flush=True)
rows = []
for c in ct.unique():
    mask = ct == c
    vc = uncorr[mask].value_counts(normalize=True)
    rows.append({"cell_type": c, "n_cells": int(mask.sum()),
                 "dominant_uncorrected_cluster": vc.index[0], "purity_pct": round(100*vc.iloc[0], 1)})
purity_df = pd.DataFrame(rows).sort_values("n_cells", ascending=False)
print(purity_df.to_string(index=False), flush=True)
purity_df.to_csv(f"{WORK}/harmony_vs_uncorrected_purity.csv", index=False)

print(f"\nDONE. ARI={ari:.4f}. peak GB: {mem():.2f}", flush=True)
