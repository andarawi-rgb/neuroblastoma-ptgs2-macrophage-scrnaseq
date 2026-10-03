#!/bin/bash
source /home/claude/nb_venv/bin/activate
cd /home/claude/nb_analysis
for f in raw/GSM*.count.csv.gz; do
  base=$(basename "$f")
  sample=$(echo "$base" | sed -E 's/GSM[0-9]+_([A-Za-z0-9]+)\.count\.csv\.gz/\1/')
  if [ -f "work/${sample}_qc_scrublet.h5ad" ]; then
    echo "SKIP $sample (already done)"
    continue
  fi
  echo "=== RUN $sample ==="
  python3 01b_load_one_sample.py "$f" >> log_01b.txt 2>&1
  if [ $? -ne 0 ]; then
    echo "FAILED $sample" >> log_01b.txt
  fi
done
echo "ALL SAMPLES PROCESSED" >> log_01b.txt
