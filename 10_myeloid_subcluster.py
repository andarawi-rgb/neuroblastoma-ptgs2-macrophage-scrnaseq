import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import gc, resource

def mem():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

WORK = "/home/claude/nb_analysis/work"

# ---- load full normalized data, subset immediately to myeloid/macrophage cells ----
adata = sc.read_h5ad(f"{WORK}/adata_norm_full.h5ad")
obs_ann = pd.read_csv(f"{WORK}/obs_annotated.csv", index_col=0)
obs_ann = obs_ann.loc[adata.obs_names]
adata.obs["cell_type"] = obs_ann["cell_type"].values
adata.obs["sample_id"] = obs_ann["sample_id"].values
adata.obs["predicted_doublet"] = obs_ann["predicted_doublet"].astype(bool).values
print(f"Loaded full normalized data: {adata.shape}. peak GB: {mem():.2f}", flush=True)

mye = adata[(adata.obs["cell_type"] == "Myeloid_Macrophage") & (~adata.obs["predicted_doublet"])].copy()
del adata
gc.collect()
print(f"Myeloid/macrophage subset (doublets removed): {mye.shape}. peak GB: {mem():.2f}", flush=True)

# ---- HVG within the myeloid subset, PCA, neighbors, Leiden at finer resolution ----
sc.pp.highly_variable_genes(mye, n_top_genes=2000, batch_key="sample_id")
mye_hvg = mye[:, mye.var["highly_variable"]].copy()
sc.pp.scale(mye_hvg, max_value=10)
sc.tl.pca(mye_hvg, n_comps=30, svd_solver="arpack", random_state=0)

import harmonypy
ho = harmonypy.run_harmony(mye_hvg.obsm["X_pca"], mye_hvg.obs, ["sample_id"], random_state=0)
Z = ho.Z_corr.T if ho.Z_corr.shape[0] != mye_hvg.n_obs else ho.Z_corr
if Z.shape[0] != mye_hvg.n_obs:
    Z = Z.T
mye_hvg.obsm["X_pca_harmony"] = np.asarray(Z)
print(f"Myeloid-specific Harmony done: {mye_hvg.obsm['X_pca_harmony'].shape}. peak GB: {mem():.2f}", flush=True)

sc.pp.neighbors(mye_hvg, use_rep="X_pca_harmony", n_neighbors=15, random_state=0)
sc.tl.leiden(mye_hvg, resolution=1.2, key_added="leiden_myeloid", flavor="igraph", n_iterations=2, random_state=0)
n_sub = mye_hvg.obs["leiden_myeloid"].nunique()
print(f"Myeloid sub-clustering: {n_sub} sub-clusters. peak GB: {mem():.2f}", flush=True)
print(mye_hvg.obs["leiden_myeloid"].value_counts().sort_index(), flush=True)

# ---- NBAtlas/Verhoeven-informed marker panel for monocyte/macrophage subtypes ----
MARKERS = {
    "Classical_Monocyte":        ["VCAN", "FCN1", "S100A8", "S100A9", "S100A12", "CD14"],
    "Patrolling_Monocyte":       ["FCGR3A", "CX3CR1", "LST1", "MS4A7"],
    "cDC_Mature_DC":             ["CD1C", "CLEC9A", "CCR7", "LAMP3", "FCER1A"],
    "LAM_like_TAM":              ["APOE", "APOC1", "TREM2", "SPP1", "FABP5", "FABP4"],
    "Tissue_Resident_FOLR2_Mac": ["FOLR2", "MRC1", "CD163", "LYVE1", "STAB1"],
    "Inflammatory_Mac":          ["CXCL2", "CXCL3", "CXCL8", "IL1B"],
    "Proliferating_Myeloid":     ["MKI67", "TOP2A", "STMN1"],
    "HLA_hi_APC_Mac":            ["HLA-DRA", "HLA-DRB1", "CD74"],
}
present = {ct: [g for g in gl if g in mye.var_names] for ct, gl in MARKERS.items()}
missing = {ct: [g for g in gl if g not in mye.var_names] for ct, gl in MARKERS.items()}
print("\nMarker genes present per subtype (using full, non-HVG-restricted gene set):", flush=True)
for ct in MARKERS:
    print(f"  {ct}: present={present[ct]}  missing={missing[ct]}", flush=True)

X = mye.X.tocsc()
def gene_mean_per_cluster(gene, cluster_series):
    gi = mye.var_names.get_loc(gene)
    col = np.asarray(X[:, gi].todense()).ravel()
    return pd.Series(col, index=mye.obs_names).groupby(cluster_series, observed=True).mean()

cluster_col = mye_hvg.obs["leiden_myeloid"].reindex(mye.obs_names)
cluster_ids = sorted(cluster_col.unique(), key=lambda x: int(x))
score_table = pd.DataFrame(index=cluster_ids)
for ct, genes in present.items():
    if not genes:
        continue
    gene_means = pd.DataFrame({g: gene_mean_per_cluster(g, cluster_col) for g in genes}).reindex(cluster_ids)
    z = (gene_means - gene_means.mean()) / gene_means.std(ddof=0).replace(0, np.nan)
    score_table[ct] = z.mean(axis=1)

subtype_call = score_table.idxmax(axis=1)  # ties broken by first-listed subtype in MARKERS dict order
print("\n=== Myeloid sub-cluster -> subtype calls (z-score idxmax; first-listed wins exact ties) ===", flush=True)
print(subtype_call.to_string(), flush=True)

mye.obs["myeloid_subtype"] = cluster_col.map(subtype_call).astype(str)

# ---- PTGS2/PTGES/PTGS1 per sub-cluster and per subtype ----
for g in ["PTGS2", "PTGES", "PTGS1"]:
    gi = mye.var_names.get_loc(g)
    mye.obs[g] = np.asarray(X[:, gi].todense()).ravel()

print("\n=== PTGS2/PTGES/PTGS1 by myeloid sub-cluster ===", flush=True)
sub_stats = mye.obs.groupby(cluster_col, observed=True).agg(
    n_cells=("PTGS2", "size"),
    PTGS2_mean=("PTGS2", "mean"),
    PTGS2_pct_pos=("PTGS2", lambda s: 100 * (s > 0).mean()),
    PTGES_mean=("PTGES", "mean"),
    PTGS1_mean=("PTGS1", "mean"),
).reindex(cluster_ids)
sub_stats["subtype_call"] = subtype_call
print(sub_stats.to_string(), flush=True)
sub_stats.to_csv(f"{WORK}/myeloid_subcluster_stats.csv")

print("\n=== PTGS2/PTGES/PTGS1 by myeloid SUBTYPE (subtypes pooled across sub-clusters) ===", flush=True)
subtype_stats = mye.obs.groupby("myeloid_subtype", observed=True).agg(
    n_cells=("PTGS2", "size"),
    PTGS2_mean=("PTGS2", "mean"),
    PTGS2_pct_pos=("PTGS2", lambda s: 100 * (s > 0).mean()),
    PTGES_mean=("PTGES", "mean"),
    PTGS1_mean=("PTGS1", "mean"),
).sort_values("n_cells", ascending=False)
print(subtype_stats.to_string(), flush=True)
subtype_stats.to_csv(f"{WORK}/myeloid_subtype_stats.csv")

# per-patient presence check (how many of 19 patients contribute to each subtype)
patient_presence = mye.obs.groupby("myeloid_subtype", observed=True)["sample_id"].nunique()
print("\nNumber of distinct patients contributing to each subtype:", flush=True)
print(patient_presence.to_string(), flush=True)

mye.obs[["sample_id", "myeloid_subtype", "PTGS2", "PTGES", "PTGS1"]].to_csv(f"{WORK}/myeloid_cells_subtyped.csv")
score_table.to_csv(f"{WORK}/myeloid_subtype_zscore_table.csv")

print(f"\nDONE myeloid sub-clustering. peak GB: {mem():.2f}", flush=True)
