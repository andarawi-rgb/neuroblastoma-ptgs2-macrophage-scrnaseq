import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.stats as stats
import gc, resource

def mem():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

WORK = "/home/claude/nb_analysis/work"

adata = sc.read_h5ad(f"{WORK}/adata_norm_full.h5ad")
obs_ann = pd.read_csv(f"{WORK}/obs_annotated.csv", index_col=0)
assert list(obs_ann.index) == list(adata.obs_names)
adata.obs["cell_type"] = obs_ann["cell_type"].values
adata.obs["predicted_doublet"] = obs_ann["predicted_doublet"].values
print(f"Loaded: {adata.shape}. peak GB: {mem():.2f}", flush=True)

GENES = ["PTGS2", "PTGES", "PTGS1"]
X = adata.X.tocsc()

def gene_vec(gene):
    gi = adata.var_names.get_loc(gene)
    col = X[:, gi]
    return np.asarray(col.todense()).ravel() if sp.issparse(col) else col

print("\n=== BASELINE (all cells, pre-doublet-removal, pre-ambient-correction) ===", flush=True)
print("Table matching manuscript Table 1 format:\n", flush=True)
rows = []
for gene in GENES:
    v = gene_vec(gene)
    adata.obs[f"_{gene}"] = v

for ct in sorted(adata.obs["cell_type"].unique(), key=lambda c: -1*(adata.obs["cell_type"]==c).sum()):
    mask = adata.obs["cell_type"] == ct
    row = {"cell_type": ct, "n_cells": int(mask.sum())}
    for gene in GENES:
        v = adata.obs.loc[mask, f"_{gene}"]
        row[f"{gene}_mean"] = round(float(v.mean()), 4)
        if gene == "PTGS2":
            row["PTGS2_pct_pos"] = round(100*float((v > 0).mean()), 2)
    rows.append(row)

table1 = pd.DataFrame(rows).sort_values("n_cells", ascending=False)
print(table1.to_string(index=False), flush=True)
table1.to_csv(f"{WORK}/baseline_table1.csv", index=False)

# ---- pseudobulk paired Wilcoxon: macrophage vs fibroblast/CAF, patient-level ----
mac = adata.obs[adata.obs["cell_type"] == "Myeloid_Macrophage"]
caf = adata.obs[adata.obs["cell_type"] == "Fibroblast_CAF"]
mac_pb = mac.groupby("sample_id", observed=True)["_PTGS2"].mean()
caf_pb = caf.groupby("sample_id", observed=True)["_PTGS2"].mean()
common = sorted(set(mac_pb.index) & set(caf_pb.index))
print(f"\nPatients with both compartments represented: {len(common)} / 19", flush=True)
diffs = mac_pb[common] - caf_pb[common]
n_concordant = int((diffs > 0).sum())
stat, p = stats.wilcoxon(mac_pb[common], caf_pb[common], alternative="two-sided")
print(f"Macrophage > CAF in {n_concordant}/{len(common)} patients.", flush=True)
print(f"Paired Wilcoxon signed-rank (two-sided): statistic={stat}, p={p:.3e}", flush=True)

pd.DataFrame({"sample_id": common, "macrophage_PTGS2": mac_pb[common].values,
              "CAF_PTGS2": caf_pb[common].values, "diff": diffs.values}).to_csv(
    f"{WORK}/baseline_pseudobulk_macrophage_vs_CAF.csv", index=False)

print(f"\nDONE baseline stats. peak GB: {mem():.2f}", flush=True)
