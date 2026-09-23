# Decision log

CRISP-DM is iterative: every change to data preparation, modeling or deployment that affects results
is recorded here (needed for Submission 2, "document significant changes and explain why").

| Date | Stage | Decision / change | Reason | Evidence (file, run, figure) | Author |
|---|---|---|---|---|---|
| 2026-09-15 | 3 Data prep | Initial pipeline: MLII lead, 0.5-40 Hz Butterworth band-pass (zero-phase), window 320 + 192 samples at 360 Hz, per-window z-score clipped to +-5 and scaled to [-1, 1], AAMI N/S/V/F mapping | Starting point; all parameters to be validated on DS1 | `configs/data.yaml` | |
| 2026-09-15 | 3 Data prep | Validation records changed from provisional [101, 106, 118, 203, 207] to [109, 201, 205, 223] | Provisional split had 1 F beat and all DS1 E beats; new split selected by `src/select_val_records.py` (~20 % of N/S/V, 29 F, records 207/208 kept in training) | `results/data_stats/split_summary.csv` | |
| 2026-09-15 | 3 Data prep | NSTDB noise records split in time: first 50 % for training augmentation, last 50 % for robustness evaluation | Prevents noise-segment leakage between training and evaluation | `configs/data.yaml`, `src/noise.py` | |
