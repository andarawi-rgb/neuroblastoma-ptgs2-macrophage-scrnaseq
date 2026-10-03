import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import scipy.sparse as sp
import gc, resource, glob, os, re

def mem():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

RAW_DIR = "/home/claude/nb_analysis/raw"
OUT_DIR = "/home/claude/nb_analysis/work"
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob(f"{RAW_DIR}/GSM*.count.csv.gz"))
print(f"Found {len(files)} sample files", flush=True)

sample_adatas = []
qc_summary = []

for fp in files:
    fname = os.path.basename(fp)
    m = re.match(r"(GSM\d+)_([A-Za-z0-9]+)\.count\.csv\.gz", fname)
    gsm, sample_id = m.group(1), m.group(2)
    print(f"\n=== Loading {sample_id} ({gsm}) === peak GB so far: {mem():.2f}", flush=True)

    # genes x cells dense CSV -> load then transpose to cells x genes, sparse immediately
    df = pd.read_csv(fp, index_col=0)
    genes = df.index.to_numpy()
    cells = df.columns.to_numpy()
    X = sp.csr_matrix(df.values.T.astype(np.float32))  # cells x genes
    del df
    gc.collect()
    print(f"  raw matrix: {X.shape[0]} cells x {X.shape[1]} genes. peak GB: {mem():.2f}", flush=True)

    a = ad.AnnData(X=X, obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=genes))
    a.var_names_make_unique()
    a.obs["sample_id"] = sample_id
    a.obs["gsm"] = gsm

    # ---- QC filter (per manuscript Methods): >=200 genes/cell, >=3 cells/gene, <20% mito ----
    n_before = a.n_obs
    sc.pp.filter_cells(a, min_genes=200)
    sc.pp.filter_genes(a, min_cells=3)
    a.var["mt"] = a.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    a = a[a.obs["pct_counts_mt"] < 20].copy()
    n_after_qc = a.n_obs

    # ---- Scrublet doublet detection (per-sample, on QC-filtered raw counts, before merging) ----
    try:
        import scrublet as scr
        scrub = scr.Scrublet(a.X, expected_doublet_rate=0.06)
        doublet_scores, predicted_doublets = scrub.scrub_doublets(verbose=False)
        if predicted_doublets is None:
            # threshold couldn't be auto-detected (common in small/clean samples) -> use scrublet's default heuristic threshold fallback
            thr = np.percentile(doublet_scores, 95) if len(doublet_scores) > 20 else 1.1
            predicted_doublets = doublet_scores > thr
        a.obs["doublet_score"] = doublet_scores
        a.obs["predicted_doublet"] = predicted_doublets.astype(bool)
    except Exception as e:
        print(f"  SCRUBLET FAILED for {sample_id}: {e}", flush=True)
        a.obs["doublet_score"] = np.nan
        a.obs["predicted_doublet"] = False

    n_doublets = int(a.obs["predicted_doublet"].sum())
    print(f"  QC: {n_before} -> {n_after_qc} cells after filter+mito. Doublets flagged: {n_doublets} ({100*n_doublets/max(n_after_qc,1):.1f}%). peak GB: {mem():.2f}", flush=True)

    qc_summary.append({
        "sample_id": sample_id, "gsm": gsm,
        "n_cells_raw": n_before, "n_cells_post_qc": n_after_qc,
        "n_genes_detected": a.n_vars,
        "n_doublets_flagged": n_doublets,
        "pct_doublets": round(100*n_doublets/max(n_after_qc,1), 2),
    })

    a.write(f"{OUT_DIR}/{sample_id}_qc_scrublet.h5ad")
    del a, X
    gc.collect()
    print(f"  saved {sample_id}_qc_scrublet.h5ad. peak GB: {mem():.2f}", flush=True)

qc_df = pd.DataFrame(qc_summary)
qc_df.to_csv(f"{OUT_DIR}/qc_scrublet_summary.csv", index=False)
print("\n=== QC + Scrublet summary across all 19 samples ===", flush=True)
print(qc_df.to_string(index=False), flush=True)
print(f"\nTOTAL cells post-QC (pre-doublet-removal): {qc_df['n_cells_post_qc'].sum()}", flush=True)
print(f"TOTAL doublets flagged: {qc_df['n_doublets_flagged'].sum()}", flush=True)
print("DONE.", flush=True)
