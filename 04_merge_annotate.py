import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import gc, resource

def mem():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

WORK = "/home/claude/nb_analysis/work"

adata = sc.read_h5ad(f"{WORK}/adata_norm_full.h5ad")
print(f"Loaded full normalized matrix: {adata.shape}. peak GB: {mem():.2f}", flush=True)

leiden = pd.read_csv(f"{WORK}/leiden_labels.csv", index_col=0)
assert list(leiden.index) == list(adata.obs_names), "obs_names order mismatch"
adata.obs["leiden"] = pd.Categorical(leiden["leiden"].astype(str).values,
                                      categories=[str(i) for i in sorted(leiden["leiden"].unique(), key=lambda x: int(x))])
print(f"Merged leiden labels. n_clusters={adata.obs['leiden'].nunique()}", flush=True)

# ---- marker panels for 8 major neuroblastoma TME cell types (per manuscript Methods) ----
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
PROSTAGLANDIN_GENES = ["PTGS2", "PTGS1", "PTGES", "PTGER2", "PTGER4"]

present = {ct: [g for g in gl if g in adata.var_names] for ct, gl in MARKERS.items()}
missing = {ct: [g for g in gl if g not in adata.var_names] for ct, gl in MARKERS.items()}
print("Marker genes present per cell type:", present, flush=True)
print("Marker genes MISSING per cell type:", {k: v for k, v in missing.items() if v}, flush=True)
print("Prostaglandin genes present:", [g for g in PROSTAGLANDIN_GENES if g in adata.var_names],
      "MISSING:", [g for g in PROSTAGLANDIN_GENES if g not in adata.var_names], flush=True)

X = adata.X
if sp.issparse(X):
    X = X.tocsc()

def gene_mean_per_cluster(gene):
    gi = adata.var_names.get_loc(gene)
    col = np.asarray(X[:, gi].todense()).ravel() if sp.issparse(X) else X[:, gi]
    return pd.Series(col, index=adata.obs_names).groupby(adata.obs["leiden"], observed=True).mean()

cluster_ids = sorted(adata.obs["leiden"].unique(), key=lambda x: int(x))
score_table = pd.DataFrame(index=cluster_ids)
for ct, genes in present.items():
    if not genes:
        continue
    gene_means = pd.DataFrame({g: gene_mean_per_cluster(g) for g in genes}).reindex(cluster_ids)
    z = (gene_means - gene_means.mean()) / gene_means.std(ddof=0).replace(0, np.nan)
    score_table[ct] = z.mean(axis=1)

print("\n=== Cell-type marker score per Leiden cluster (z-scored mean expression) ===", flush=True)
print(score_table.round(2).to_string(), flush=True)

cluster_call = score_table.idxmax(axis=1)
print("\n=== Provisional cell-type call per cluster ===", flush=True)
print(cluster_call, flush=True)

adata.obs["cell_type"] = adata.obs["leiden"].map(cluster_call).astype(str)
print("\nCell-type composition (n cells):", flush=True)
print(adata.obs["cell_type"].value_counts(), flush=True)

# raw marker means per cluster, for write-up / sanity check
raw_marker_means = pd.DataFrame(index=cluster_ids)
for ct, genes in present.items():
    for g in genes:
        raw_marker_means[g] = gene_mean_per_cluster(g).reindex(cluster_ids)
for g in PROSTAGLANDIN_GENES:
    if g in adata.var_names:
        raw_marker_means[g] = gene_mean_per_cluster(g).reindex(cluster_ids)
raw_marker_means.to_csv(f"{WORK}/cluster_marker_means.csv")

score_table.to_csv(f"{WORK}/cluster_celltype_scores.csv")
cluster_call.to_csv(f"{WORK}/cluster_celltype_calls.csv")
adata.obs[["sample_id","gsm","leiden","cell_type","doublet_score","predicted_doublet",
           "pct_counts_mt","n_genes_by_counts"]].to_csv(f"{WORK}/obs_annotated.csv")

print("\nSaved cluster_marker_means.csv, cluster_celltype_scores.csv, cluster_celltype_calls.csv, obs_annotated.csv.", flush=True)
print(f"DONE annotate. peak GB: {mem():.2f}", flush=True)
