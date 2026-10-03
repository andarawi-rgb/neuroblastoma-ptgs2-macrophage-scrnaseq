import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import scipy.sparse as sp
import gc, resource, glob

def mem():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

WORK = "/home/claude/nb_analysis/work"
OUT = "/home/claude/nb_analysis/work"

files = sorted(glob.glob(f"{WORK}/*_qc_scrublet.h5ad"))
print(f"Merging {len(files)} samples. peak GB: {mem():.2f}", flush=True)

adatas = []
for fp in files:
    a = sc.read_h5ad(fp)
    a.obs_names = a.obs["sample_id"].astype(str) + "_" + a.obs_names.astype(str)
    adatas.append(a)

merged = ad.concat(adatas, join="outer", fill_value=0)
del adatas
gc.collect()
merged.X = sp.csr_matrix(merged.X)
print(f"Merged (pre-doublet-removal, BASELINE = matches original pipeline): "
      f"{merged.shape}. peak GB: {mem():.2f}", flush=True)
merged.write(f"{OUT}/adata_merged_all_cells.h5ad")

n_doublets = int(merged.obs["predicted_doublet"].sum())
print(f"Doublets flagged (p95 per-sample threshold): {n_doublets} / {merged.n_obs} "
      f"({100*n_doublets/merged.n_obs:.2f}%)", flush=True)

merged.obs[["sample_id","gsm","doublet_score","predicted_doublet","predicted_doublet_auto",
            "pct_counts_mt","n_genes_by_counts"]].to_csv(f"{OUT}/merged_obs_meta.csv")
print("DONE merge.", flush=True)
