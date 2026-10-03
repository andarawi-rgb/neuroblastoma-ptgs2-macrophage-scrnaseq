import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gc, resource

def mem():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

WORK = "/home/claude/nb_analysis/work"

adata = sc.read_h5ad(f"{WORK}/adata_norm_full.h5ad")
obs_ann = pd.read_csv(f"{WORK}/obs_annotated.csv", index_col=0).loc[adata.obs_names]
adata.obs["cell_type"] = obs_ann["cell_type"].values
print(f"Loaded: {adata.shape}. peak GB: {mem():.2f}", flush=True)

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
cell_type_order = list(MARKERS.keys())
all_genes = [g for genes in MARKERS.values() for g in genes]

X = adata.X.tocsc()
rows = []
for ct in cell_type_order:
    mask = (adata.obs["cell_type"] == ct).values
    for panel_ct, genes in MARKERS.items():
        for g in genes:
            gi = adata.var_names.get_loc(g)
            col = np.asarray(X[mask, gi].todense()).ravel()
            rows.append({"cell_type": ct, "gene": g, "panel_for": panel_ct,
                         "mean_expr": col.mean(), "pct_pos": 100 * (col > 0).mean()})
df = pd.DataFrame(rows)
df.to_csv(f"{WORK}/dotplot_data.csv", index=False)
print(f"Computed dot-plot data: {df.shape}. peak GB: {mem():.2f}", flush=True)
del adata, X
gc.collect()

# ---- build the dot plot: rows = cell types, cols = marker genes (grouped by intended panel) ----
gene_cols = all_genes
n_rows, n_cols = len(cell_type_order), len(gene_cols)
mean_mat = np.zeros((n_rows, n_cols))
pct_mat = np.zeros((n_rows, n_cols))
for i, ct in enumerate(cell_type_order):
    for j, g in enumerate(gene_cols):
        sub = df[(df["cell_type"] == ct) & (df["gene"] == g)]
        mean_mat[i, j] = sub["mean_expr"].values[0]
        pct_mat[i, j] = sub["pct_pos"].values[0]

# z-score mean expression per gene (column) for color scale, as used in the actual annotation procedure
mean_z = (mean_mat - mean_mat.mean(axis=0, keepdims=True)) / (mean_mat.std(axis=0, keepdims=True) + 1e-9)

fig, ax = plt.subplots(figsize=(0.42 * n_cols + 2, 0.42 * n_rows + 1.5))
xs, ys = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
sizes = 8 + (pct_mat / 100.0) * 220
sc_plot = ax.scatter(xs.ravel(), ys.ravel(), s=sizes.ravel(), c=mean_z.ravel(),
                      cmap="RdBu_r", vmin=-2, vmax=2, edgecolors="black", linewidths=0.3)

# panel boundary separators
boundary = 0
for ct in cell_type_order:
    n = len(MARKERS[ct])
    if boundary > 0:
        ax.axvline(boundary - 0.5, color="grey", linewidth=0.6, linestyle="--")
    boundary += n

ax.set_xticks(np.arange(n_cols))
ax.set_xticklabels(gene_cols, rotation=90, fontsize=7, fontstyle="italic")
ax.set_yticks(np.arange(n_rows))
ax.set_yticklabels([c.replace("_", "/") for c in cell_type_order], fontsize=8)
ax.set_xlim(-0.5, n_cols - 0.5)
ax.set_ylim(-0.5, n_rows - 0.5)
ax.invert_yaxis()
ax.set_title("Marker-gene panel expression across the 8 annotated cell types\n(dot size = % cells positive; color = column-wise z-scored mean expression)",
             fontsize=9)

cbar = plt.colorbar(sc_plot, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("z-scored mean expression", fontsize=7)
cbar.ax.tick_params(labelsize=6)

for pct, label in [(10, "10%"), (50, "50%"), (100, "100%")]:
    ax.scatter([], [], s=8 + (pct/100.0)*220, c="grey", edgecolors="black", linewidths=0.3, label=label)
ax.legend(title="% positive", loc="upper left", bbox_to_anchor=(1.12, 1.0), fontsize=6, title_fontsize=7, frameon=False)

plt.tight_layout()
plt.savefig(f"{WORK}/Figure_S_annotation_dotplot.png", dpi=300, bbox_inches="tight")
print(f"Saved dot plot. peak GB: {mem():.2f}", flush=True)
