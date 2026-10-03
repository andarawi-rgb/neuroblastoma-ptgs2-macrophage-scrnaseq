import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import gc, resource, time

def mem():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

WORK = "/home/claude/nb_analysis/work"

adata = sc.read_h5ad(f"{WORK}/adata_merged_all_cells.h5ad")  # RAW counts
print(f"Loaded raw counts: {adata.shape}. peak GB: {mem():.2f}", flush=True)

hvg_genes = sc.read_h5ad(f"{WORK}/adata_hvg_only.h5ad").var_names.tolist()
for g in ["PTGS2", "PTGES", "PTGS1"]:
    if g not in hvg_genes:
        hvg_genes.append(g)
gene_list = [g for g in hvg_genes if g in adata.var_names]
adata_s = adata[:, gene_list].copy()
sample_ids = adata_s.obs["sample_id"].astype(str).values
del adata
gc.collect()
print(f"Working subset for scAR: {adata_s.shape} (HVG + PTGS2/PTGES/PTGS1). peak GB: {mem():.2f}", flush=True)

import scar
t0 = time.time()
m = scar.model(adata_s.X.toarray(), ambient_profile=None, batch_key=None, sparsity=0.9, verbose=True)
print(f"Model init: {time.time()-t0:.1f}s. peak GB: {mem():.2f}", flush=True)

EPOCHS = 100
t0 = time.time()
m.train(epochs=EPOCHS, verbose=True)
print(f"Training {EPOCHS} epochs took {time.time()-t0:.1f}s. peak GB: {mem():.2f}", flush=True)

t0 = time.time()
m.inference()
print(f"Inference took {time.time()-t0:.1f}s. peak GB: {mem():.2f}", flush=True)

denoised = m.native_counts
if hasattr(denoised, "toarray"):
    denoised = denoised.toarray()
denoised = np.asarray(denoised)
print(f"Denoised counts shape: {denoised.shape}. peak GB: {mem():.2f}", flush=True)

np.save(f"{WORK}/scar_denoised_counts.npy", denoised.astype(np.float32))
pd.Series(gene_list).to_csv(f"{WORK}/scar_gene_list.csv", index=False, header=["gene"])
pd.Series(adata_s.obs_names).to_csv(f"{WORK}/scar_cell_order.csv", index=False, header=["cell"])
print("Saved scar_denoised_counts.npy, scar_gene_list.csv, scar_cell_order.csv. DONE.", flush=True)
