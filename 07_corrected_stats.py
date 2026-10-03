import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.stats as stats
import gc, resource

def mem():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

WORK = "/home/claude/nb_analysis/work"

# ---- load scAR-denoised counts (HVG + PTGS2/PTGES/PTGS1), re-normalize per cell ----
denoised = np.load(f"{WORK}/scar_denoised_counts.npy")
genes = pd.read_csv(f"{WORK}/scar_gene_list.csv")["gene"].tolist()
cells = pd.read_csv(f"{WORK}/scar_cell_order.csv")["cell"].tolist()
print(f"Loaded scAR-denoised matrix: {denoised.shape}. peak GB: {mem():.2f}", flush=True)

# library-size normalize the denoised counts (same convention as original pipeline: target_sum=1e4, log1p)
# use the denoised matrix's own row sums as the per-cell size factor (post ambient-removal library size)
libsize = denoised.sum(axis=1, keepdims=True)
libsize[libsize == 0] = 1
norm = denoised / libsize * 1e4
lognorm = np.log1p(norm)
gene_idx = {g: i for i, g in enumerate(genes)}

obs_ann = pd.read_csv(f"{WORK}/obs_annotated.csv", index_col=0)
obs_ann = obs_ann.loc[cells]  # align to scAR cell order
assert list(obs_ann.index) == cells

sample_id = obs_ann["sample_id"].values
cell_type = obs_ann["cell_type"].values
predicted_doublet = obs_ann["predicted_doublet"].astype(bool).values

print(f"\n=== AMBIENT-RNA-CORRECTED (scAR) + DOUBLET-REMOVED analysis ===", flush=True)
n_before = len(cells)
keep = ~predicted_doublet
n_after = keep.sum()
print(f"Removing {n_before - n_after} flagged doublets ({100*(n_before-n_after)/n_before:.2f}%). "
      f"{n_after} cells remain.", flush=True)

df = pd.DataFrame({
    "sample_id": sample_id[keep],
    "cell_type": cell_type[keep],
    "PTGS2": lognorm[keep, gene_idx["PTGS2"]],
    "PTGES": lognorm[keep, gene_idx["PTGES"]],
    "PTGS1": lognorm[keep, gene_idx["PTGS1"]],
})

print("\n=== Table 1 equivalent, AFTER scAR ambient correction + doublet removal ===", flush=True)
rows = []
for ct, sub in df.groupby("cell_type", observed=True):
    rows.append({
        "cell_type": ct, "n_cells": len(sub),
        "PTGS2_mean": round(sub["PTGS2"].mean(), 4),
        "PTGS2_pct_pos": round(100*(sub["PTGS2"] > 0).mean(), 2),
        "PTGES_mean": round(sub["PTGES"].mean(), 4),
        "PTGS1_mean": round(sub["PTGS1"].mean(), 4),
    })
table_corrected = pd.DataFrame(rows).sort_values("n_cells", ascending=False)
print(table_corrected.to_string(index=False), flush=True)
table_corrected.to_csv(f"{WORK}/corrected_table1.csv", index=False)

mac = df[df["cell_type"] == "Myeloid_Macrophage"]
caf = df[df["cell_type"] == "Fibroblast_CAF"]
mac_pb = mac.groupby("sample_id", observed=True)["PTGS2"].mean()
caf_pb = caf.groupby("sample_id", observed=True)["PTGS2"].mean()
common = sorted(set(mac_pb.index) & set(caf_pb.index))
print(f"\nPatients with both compartments (post doublet-removal): {len(common)} / 19", flush=True)

diffs = mac_pb[common] - caf_pb[common]
n_pos = int((diffs > 0).sum())
n_neg = int((diffs < 0).sum())
n_tie = int((diffs == 0).sum())
stat, p = stats.wilcoxon(mac_pb[common], caf_pb[common], alternative="two-sided")
print(f"Macrophage > CAF: {n_pos}, Macrophage < CAF: {n_neg}, tied: {n_tie} (of {len(common)})", flush=True)
print(f"Paired Wilcoxon signed-rank (two-sided): statistic={stat}, p={p:.3e}", flush=True)

# effect size: matched-pairs rank-biserial correlation, r = (W+ - W-) / (W+ + W-) using signed ranks
d = (mac_pb[common] - caf_pb[common]).values
d_nz = d[d != 0]
ranks = stats.rankdata(np.abs(d_nz))
Wpos = ranks[d_nz > 0].sum()
Wneg = ranks[d_nz < 0].sum()
rank_biserial = (Wpos - Wneg) / (Wpos + Wneg) if (Wpos+Wneg) > 0 else np.nan
# Hodges-Lehmann estimator (median of pairwise Walsh averages)
walsh = np.array([(d[i]+d[j])/2 for i in range(len(d)) for j in range(i, len(d))])
hl_estimate = np.median(walsh)
print(f"Effect size: rank-biserial r = {rank_biserial:.3f}, Hodges-Lehmann estimate of median diff = {hl_estimate:.4f}", flush=True)

pd.DataFrame({"sample_id": common, "macrophage_PTGS2": mac_pb[common].values,
              "CAF_PTGS2": caf_pb[common].values, "diff": diffs.values}).to_csv(
    f"{WORK}/corrected_pseudobulk_macrophage_vs_CAF.csv", index=False)

# ---- side-by-side comparison with baseline ----
baseline_table = pd.read_csv(f"{WORK}/baseline_table1.csv")
print("\n=== BASELINE vs CORRECTED comparison (macrophage & CAF rows) ===", flush=True)
for ct in ["Myeloid_Macrophage", "Fibroblast_CAF"]:
    b = baseline_table[baseline_table["cell_type"] == ct].iloc[0]
    c = table_corrected[table_corrected["cell_type"] == ct].iloc[0]
    print(f"{ct}: baseline PTGS2_mean={b['PTGS2_mean']}, pct_pos={b['PTGS2_pct_pos']}  ->  "
          f"corrected PTGS2_mean={c['PTGS2_mean']}, pct_pos={c['PTGS2_pct_pos']}", flush=True)

print(f"\nDONE corrected stats. peak GB: {mem():.2f}", flush=True)
