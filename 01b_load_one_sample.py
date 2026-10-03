import sys, os, re, gc, resource
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import scipy.sparse as sp

def mem():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

fp = sys.argv[1]
OUT_DIR = "/home/claude/nb_analysis/work"
os.makedirs(OUT_DIR, exist_ok=True)

fname = os.path.basename(fp)
m = re.match(r"(GSM\d+)_([A-Za-z0-9]+)\.count\.csv\.gz", fname)
gsm, sample_id = m.group(1), m.group(2)
print(f"=== {sample_id} ({gsm}) === start peak GB: {mem():.2f}", flush=True)

# ---- memory-safe chunked read: genes x cells CSV -> sparse, never materialize a full dense array ----
CHUNK = 1500
header_cols = pd.read_csv(fp, nrows=0).columns.tolist()
gene_col = header_cols[0]
value_cols = header_cols[1:]
dtype_map = {c: np.float32 for c in value_cols}
dtype_map[gene_col] = str

reader = pd.read_csv(fp, index_col=0, chunksize=CHUNK, dtype=dtype_map)
gene_chunks = []
gene_names = []
cells = None
for chunk in reader:
    if cells is None:
        cells = chunk.columns.to_numpy()
    gene_names.extend(chunk.index.tolist())
    gene_chunks.append(sp.csr_matrix(chunk.to_numpy(dtype=np.float32, copy=False)))
    del chunk
del reader
gc.collect()

X_gxc = sp.vstack(gene_chunks, format="csr")
del gene_chunks
gc.collect()
X = X_gxc.T.tocsr()  # cells x genes
del X_gxc
gc.collect()
print(f"  parsed sparse matrix: {X.shape[0]} cells x {X.shape[1]} genes. peak GB: {mem():.2f}", flush=True)

genes = np.array(gene_names)
a = ad.AnnData(X=X, obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=genes))
a.var_names_make_unique()
a.obs["sample_id"] = sample_id
a.obs["gsm"] = gsm
del X
gc.collect()

n_before = a.n_obs
sc.pp.filter_cells(a, min_genes=200)
sc.pp.filter_genes(a, min_cells=3)
a.var["mt"] = a.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
a = a[a.obs["pct_counts_mt"] < 20].copy()
n_after_qc = a.n_obs
print(f"  QC: {n_before} -> {n_after_qc} cells. peak GB: {mem():.2f}", flush=True)

try:
    import scrublet as scr
    scrub = scr.Scrublet(a.X, expected_doublet_rate=0.06)
    doublet_scores, predicted_doublets = scrub.scrub_doublets(verbose=False)
    if predicted_doublets is None:
        thr = np.percentile(doublet_scores, 95) if len(doublet_scores) > 20 else 1.1
        predicted_doublets = doublet_scores > thr
    a.obs["doublet_score"] = doublet_scores
    a.obs["predicted_doublet"] = np.asarray(predicted_doublets).astype(bool)
except Exception as e:
    print(f"  SCRUBLET FAILED for {sample_id}: {e}", flush=True)
    a.obs["doublet_score"] = np.nan
    a.obs["predicted_doublet"] = False

n_doublets = int(a.obs["predicted_doublet"].sum())
print(f"  Doublets flagged: {n_doublets} ({100*n_doublets/max(n_after_qc,1):.1f}%). peak GB: {mem():.2f}", flush=True)

a.write(f"{OUT_DIR}/{sample_id}_qc_scrublet.h5ad")

summary_row = pd.DataFrame([{
    "sample_id": sample_id, "gsm": gsm,
    "n_cells_raw": n_before, "n_cells_post_qc": n_after_qc,
    "n_genes_detected": a.n_vars,
    "n_doublets_flagged": n_doublets,
    "pct_doublets": round(100*n_doublets/max(n_after_qc,1), 2),
}])
summary_path = f"{OUT_DIR}/qc_scrublet_summary.csv"
summary_row.to_csv(summary_path, mode="a", header=not os.path.exists(summary_path), index=False)
print(f"DONE {sample_id}. final peak GB: {mem():.2f}", flush=True)
