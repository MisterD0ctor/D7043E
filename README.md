# Edge-Ready ECG Classification using CRISP-DM

D7043E course project (LTU). Four-class heartbeat classification (**N**, **S**, **V**, **F**) on the
MIT-BIH Arrhythmia Database with a compact 1D CNN designed for the Analog Devices **MAX78002**
Edge-AI microcontroller, including INT8 quantization, noise-robustness evaluation and software-level
synthesis with the ai8x toolchain.

> This is a course project studying the data-mining and Edge-AI development process.
> It is **not** a medical device and is **not** clinically validated. No MAX78002 hardware is used;
> all deployment results are software-verified only.

**Group XX:** _name 1_, _name 2_, _name 3_, _name 4_

## Repository structure

```
├── README.md, requirements.txt
├── configs/
│   ├── data.yaml            # every data-preparation parameter (splits, lead, filter, window, labels, noise)
│   └── train.yaml           # FP32 model, training, class-imbalance and augmentation settings
├── src/
│   ├── download_data.py     # PhysioNet download (MIT-BIH + Noise Stress Test DB)
│   ├── prepare_data.py      # beat extraction, labelling, normalization, statistics        (stage 3)
│   ├── select_val_records.py  # reproducible record-level validation split inside DS1
│   ├── features.py          # 79 signal measures per beat for the EDA (amplitude, shape, QRS, rhythm, spectrum, ...)
│   ├── explore_*.py         # exploratory model comparison on DS1 only (see docs/exploration_notes.md)
│   ├── dataset.py           # PyTorch dataset, augmentation (train only), class weights
│   ├── model.py             # compact MAX78002-compatible 1D CNN                             (stage 4)
│   ├── train.py             # training with record-level validation and early stopping
│   ├── evaluate.py          # accuracy, macro F1, per-class metrics, confusion matrix        (stage 5)
│   ├── noise_test.py        # robustness under NSTDB noise at 12 / 6 / 0 dB SNR
│   ├── noise.py             # NSTDB loading and SNR scaling helpers
│   └── config.py            # config loading, label mapping, split leakage checks
├── notebooks/EDA.ipynb      # data understanding                                             (stage 2)
├── results/
│   ├── data_stats/          # generated split / record / annotation statistics (committed)
│   ├── figures/             # EDA figures used in the report
│   ├── metrics.csv, confusion_matrix_*, noise_results_*
│   └── runs/                # training checkpoints and histories (git-ignored)
├── max78002/                # ai8x model, dataset loader, network YAML, quantization + synthesis output (stage 6)
├── report/report.md         # CRISP-DM report draft (exported to report.pdf)
└── docs/decision_log.md     # CRISP-DM iterations: what changed and why
```

## Setup

Python **3.11** (3.11.8 recommended, same as the ai8x toolchain).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # optional: NVIDIA GPU
pip install -r requirements.txt
```

All commands below are run from the repository root.

## Reproducing the data preparation (Submission 1)

### 1. Download the data

```bash
python src/download_data.py
```

Downloads into `data/raw/` (~110 MB, git-ignored; do not commit or submit raw data):

- MIT-BIH Arrhythmia Database v1.0.0 - https://physionet.org/content/mitdb/1.0.0/
- MIT-BIH Noise Stress Test Database v1.0.0 - https://physionet.org/content/nstdb/1.0.0/

Manual alternative: download the ZIP files from the PhysioNet pages and extract them so that
`data/raw/mitdb/100.hea` and `data/raw/nstdb/em.hea` exist.

### 2. Build the datasets

```bash
python src/prepare_data.py
```

Processing per record, in this order (parameters in `configs/data.yaml`):

1. Load lead **MLII** by signal name (360 Hz, mV)
2. 4th-order Butterworth band-pass 0.5-40 Hz, zero-phase, on the whole record
3. Window of 320 samples before and 192 after each annotated beat -> 512 samples (1.42 s)
4. Map annotation symbols to N/S/V/F; other annotations are excluded and counted
5. Per-window z-score, clip to +-5, divide by 5 -> input range [-1, 1], shape `(1, 512)`

Outputs:

| File | Content |
|---|---|
| `data/processed/{train,val,test,excluded}.npz` | `X` (n, 1, 512) float32, `y` (0=N, 1=S, 2=V, 3=F), `record`, `sample`, `symbol`, RR intervals |
| `results/data_stats/split_summary.csv` | beats per class per split |
| `results/data_stats/cv_folds.csv` | cross-validation folds over the development records with class counts |
| `results/data_stats/record_stats.csv` | per-record class counts, exclusions, amplitude and clipping statistics |
| `results/data_stats/annotation_mapping.csv` | every annotation symbol -> final class or exclusion reason, with counts |
| `results/data_stats/prep_manifest.json` | full config, software versions, seed, git commit, SHA-256 of the outputs |

Re-running with the same config and package versions reproduces identical `.npz` checksums.

### 3. Exploratory data analysis

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/EDA.ipynb
```

The notebook analyses DS1 only (DS2 enters through record headers and class counts). It writes the figures
`results/figures/eda_*.png` and two statistics files: `results/data_stats/data_quality.csv` (per DS1 record) and
`results/data_stats/feature_separability.csv` (class separability of every signal measure in `src/features.py`).

## Data splits and leakage rules

Inter-patient partition (de Chazal et al., 2004):

| Split | Records |
|---|---|
| DS1 (development) | 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230 |
| DS2 (final test) | 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234 |

| Validation (from DS1) | 109, 205, 223 |
| Excluded (from DS1) | 201 |

The validation records are set in `configs/data.yaml` (`splits.val`). They were originally chosen with
`python src/select_val_records.py` together with record 201, while records 208 (90 % of the DS1 fusion beats) and
207 (all ventricular escape beats) stay in training. With record 201 excluded, 76 S and 27 F beats remain for
validation, so model selection should rest on record-grouped cross-validation over the development records.

- Splits are always **by record**; `config.split_records()` asserts that no record is shared.
- DS2 is only used for the final evaluation: `evaluate.py` and `noise_test.py` refuse `--split test` without `--final`.
- Model and hyperparameter selection use leave-one-record-out cross-validation over the 21 development records (`cross_validation` in `configs/data.yaml`, folds in `results/data_stats/cv_folds.csv`); the validation records only monitor a training run.
- NSTDB noise records are split in time: first half for augmentation, second half for robustness evaluation.
- Records 201 and 202 come from the same subject (DS1 and DS2 respectively). Record 201 is therefore not used for training, validation or model selection (`splits.exclude` in `configs/data.yaml`); it is still prepared (`excluded.npz`) and described in the EDA.

## Modeling, evaluation and deployment (Submission 2)

```bash
python src/model.py                                              # architecture + parameter count
python src/train.py --config configs/train.yaml                  # -> results/runs/<run_name>/best.pt
python src/evaluate.py --checkpoint results/runs/ecgnet_baseline/best.pt --split val
python src/noise_test.py --checkpoint results/runs/ecgnet_baseline/best.pt --split val

# final evaluation, once the model is frozen
python src/evaluate.py --checkpoint results/runs/<final>/best.pt --split test --final
python src/noise_test.py --checkpoint results/runs/<final>/best.pt --split test --final
```

Quantization-aware training, INT8 evaluation and MAX78002 synthesis: see [max78002/README.md](max78002/README.md).

## Submission package

The final ZIP (`GroupXX_ECG_Project/`) is this repository without `data/`, `.venv/` and `results/runs/`,
plus `report.pdf` exported from `report/report.md`.
