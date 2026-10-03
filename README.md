# Single-cell PTGS2 (COX-2) / macrophage analysis in neuroblastoma

Analysis code accompanying the manuscript: *"Single-cell resolution reveals macrophages, not mesenchymal fibroblasts, as the dominant cellular source of PTGS2 (COX-2) transcript in neuroblastoma"* (Mustafa AEM, Andarawi MO, Elhag OO, Ali NMT, Bin Fardan ZH; manuscript in preparation, target journal BMC Cancer).

## Data source

Raw, publicly deposited single-cell count matrices for 19 neuroblastoma patients, GEO accession **[GSE147766](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE147766)** (Verhoeven et al., *Cell Reports Medicine* 2022). This repository does not redistribute the raw data — download the `GSM*.count.csv.gz` sample files from GEO into a local `raw/` directory before running the pipeline.

Independent bulk-tumor validation used the **TARGET-NBL** cohort (n = 142) via [cBioPortal](https://www.cbioportal.org/study/summary?id=nbl_target_2018_pub), study ID `nbl_target_2018_pub`.

## Pipeline

Scripts are numbered in the order they were run. Each reads/writes `.h5ad` / `.csv` / `.npy` intermediate files under a local `work/` directory (created automatically). Paths are set as constants (`RAW_DIR`, `WORK`) near the top of each script — edit these for your own environment before running.

| Script | Purpose |
|---|---|
| `01_load_qc_scrublet.py`, `01b_load_one_sample.py` | Load raw per-sample count matrices, QC filtering (min genes/cell, min cells/gene, mitochondrial fraction), per-sample Scrublet doublet detection |
| `02_merge_baseline.py` | Merge all QC'd, doublet-flagged samples into one AnnData object |
| `03_cluster_annotate.py`, `03a_normalize_hvg.py`, `03b_scale_pca_harmony_leiden.py` | Normalization, HVG selection, PCA, Harmony batch integration, Leiden clustering |
| `04_merge_annotate.py` | Marker-panel z-score cell-type annotation (8 major cell types) |
| `05_baseline_stats.py` | Primary compartment-restriction statistics (cell-level and patient-level pseudobulk, Wilcoxon signed-rank, rank-biserial effect size) |
| `06_scar_pilot.py`, `06_scar_full.py` | Ambient-RNA correction sensitivity analysis (scAR) |
| `07_corrected_stats.py` | Primary statistics recomputed on the ambient-RNA/doublet-corrected data |
| `08_harmony_ari.py`, `09_uncorrected_annotate_ari.py` | Batch-correction-choice robustness check (adjusted Rand index, Harmony-corrected vs. uncorrected clustering) |
| `10_myeloid_subcluster.py` | Myeloid/macrophage sub-clustering into candidate TAM/monocyte subtypes |
| `11_dotplot.py` | Marker-panel dot-plot figure generation |
| `run_all_samples.sh` | Driver script for per-sample loading (`01b_load_one_sample.py`) across all GEO samples |

INRG risk-group statistics (per-patient macrophage *PTGS2* vs. clinical risk group, matched to Verhoeven et al. Table S1) and the TARGET-NBL bulk validation were computed in companion scripts developed during manuscript revision; see the manuscript's Methods section for the full statistical specification.

## Environment

```
pip install -r requirements.txt
```

Random seeds: all stochastic steps (PCA, neighbor-graph construction, Leiden clustering, Harmony, Scrublet) use each library's own default `random_state = 0`, left unmodified throughout, so results are reproducible given identical input data and package versions. The one exception is scAR (ambient-RNA denoising), whose public API does not expose a fixed seed — see the manuscript's Limitations for this caveat.

## Citation

If you use this code, please cite the manuscript (citation to be added on publication) and the original data source (Verhoeven et al., *Cell Reports Medicine* 2022; GEO: GSE147766).

## License

MIT — see `LICENSE`.

## Use of AI tools

Analysis code was developed with the assistance of Claude (Anthropic), under the authors' direct supervision and independent verification of intermediate outputs and statistical results. See the manuscript's "Use of AI tools" declaration for full detail.
