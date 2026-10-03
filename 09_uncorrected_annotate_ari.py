import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score
import gc, resource

def mem():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

WORK = "/home/claude/nb_analysis/work"

adata = sc.read_h5ad(f"{WORK}/adata_norm_full.h5ad")
uncorr = pd.read_csv(f"{WORK}/leiden_uncorrected_labels.csv", index_col=0)
assert list(uncorr.index) == list(adata.obs_names)
adata.obs["leiden_uncorrected"] = pd.Categorical(uncorr["leiden_uncorrected"].astype(str).values)
print(f"Loaded + merged uncorrected leiden: {adata.shape}. peak GB: {mem():.2f}", flush=True)

MARKERS = {
    "Fibroblast_CAF":    ["DCN", "PDGFRA", "PDGFRB", "COL1A1", "COL3A1", "LUM"],
    "Endothelial":       ["PECAM1", "VWF", "CDH5", "CLDN5"],
    "Myeloid_Macrophage": ["CD68", "CD14", "ITGAM", "LYZ", "AIF1", "C1QA", "C1QB", "CSF1R", "FCGR3A"],
    "T_cell":            ["CD3D", "CD3E", "CD2", "CD8A", "CD4", "IL7R"],
    "B_cell":            ["CD79A", "MS4A1", "CD19", "MZB1"],
    "NK":                ["NKG7", "GNLY", "KLRD1", "NCAM1", "KLRF1"],
    "Neuroblast_Tumor":  ["PHOX2B", "TH", "DBH", "MYCN", "GATA3", "CHGA"],
    "Schwann_SCP":       ["SOX10", "S100B", "PLP1", "MPZ"],
}
present = {ct: [g for g in gl if g in adata.var_names] for ct, gl in MARKERS.items()}

X = adata.X.tocsc()
def gene_mean_per_cluster(gene, cluster_col):
    gi = adata.var_names.get_loc(gene)
    col = np.asarray(X[:, gi].todense()).ravel()
    return pd.Series(col, index=adata.obs_names).groupby(adata.obs[cluster_col], observed=True).mean()

cluster_ids = sorted(adata.obs["leiden_uncorrected"].unique(), key=lambda x: int(x))
score_table = pd.DataFrame(index=cluster_ids)
for ct, genes in present.items():
    gene_means = pd.DataFrame({g: gene_mean_per_cluster(g, "leiden_uncorrected") for g in genes}).reindex(cluster_ids)
    z = (gene_means - gene_means.mean()) / gene_means.std(ddof=0).replace(0, np.nan)
    score_table[ct] = z.mean(axis=1)

cluster_call_uncorr = score_table.idxmax(axis=1)
adata.obs["cell_type_uncorrected"] = adata.obs["leiden_uncorrected"].map(cluster_call_uncorr).astype(str)
print("\nCell-type composition (UNCORRECTED, PCA-only clustering):", flush=True)
print(adata.obs["cell_type_uncorrected"].value_counts(), flush=True)

obs_ann = pd.read_csv(f"{WORK}/obs_annotated.csv", index_col=0).loc[adata.obs_names]
harmony_ct = obs_ann["cell_type"]
uncorr_ct = adata.obs["cell_type_uncorrected"]

ari_celltype = adjusted_rand_score(harmony_ct.astype(str), uncorr_ct.astype(str))
agreement_pct = 100 * (harmony_ct.values == uncorr_ct.values).mean()
print(f"\n=== CELL-TYPE-level concordance, Harmony-corrected vs uncorrected clustering ===", flush=True)
print(f"Adjusted Rand Index (cell-type labels): {ari_celltype:.4f}", flush=True)
print(f"Direct label agreement: {agreement_pct:.2f}% of cells get the same cell-type call either way", flush=True)

ct_tab = pd.crosstab(harmony_ct, uncorr_ct)
print("\nCross-tabulation (rows=Harmony-corrected call, cols=uncorrected call):", flush=True)
print(ct_tab.to_string(), flush=True)
ct_tab.to_csv(f"{WORK}/celltype_crosstab_harmony_vs_uncorrected.csv")

for ct in ["Myeloid_Macrophage", "Fibroblast_CAF"]:
    n_h = (harmony_ct == ct).sum()
    n_u = (uncorr_ct == ct).sum()
    n_both = ((harmony_ct == ct) & (uncorr_ct == ct)).sum()
    print(f"{ct}: n(Harmony)={n_h}, n(uncorrected)={n_u}, n(both agree)={n_both} "
          f"({100*n_both/n_h:.1f}% of Harmony-called cells retain the same call uncorrected)", flush=True)

print(f"\nDONE. peak GB: {mem():.2f}", flush=True)
