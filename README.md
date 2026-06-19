# network-traffic-transformer-lstm

> **Network Traffic Classification using a Transformer-Enhanced LSTM model**
> Built with TensorFlow/Keras · Evaluated on two real Kaggle datasets ·
> Benchmarked against a plain-LSTM baseline

---

## Project Overview

This project trains a hybrid deep-learning model — combining an **LSTM** layer
with a **Multi-Head Self-Attention (Transformer)** block — to classify network
traffic. Two public datasets are used for benchmarking, and the proposed
model is compared against a **simple baseline (plain stacked-LSTM, no
attention)**, as required by the project objectives.

| # | Dataset | File in repo | Task |
|---|---------|---------------|------|
| 1 | CIC-Darknet2020 | `dataset/cicdarknet2020.parquet` | Tor/VPN-style traffic classes |
| 2 | UNSW-NB15 | `dataset/unsw_nb15.csv` | Normal vs. Attack (binary) |

See `dataset/dataset_link.txt` for source links.

---

## Architecture

**Proposed model — Transformer-Enhanced LSTM** (`model_type=transformer_lstm`):
```
Input (samples, 1, features)
  -> LSTM (128 units, return_sequences=True)
  -> MultiHeadAttention (4 heads, key_dim=32)
  -> Residual Add + LayerNormalization
  -> GlobalAveragePooling1D
  -> Dropout(0.3) -> Dense(128, relu) -> Dropout(0.3)
  -> Output: Dense(1, sigmoid) [binary] or Dense(N, softmax) [multi-class]
```

**Baseline model — plain stacked LSTM, no attention** (`model_type=baseline_lstm`):
```
Input (samples, 1, features)
  -> LSTM(64, return_sequences=True) -> LSTM(64)
  -> Dense(64, relu) -> Dropout(0.3)
  -> Output: Dense(1, sigmoid) [binary] or Dense(N, softmax) [multi-class]
```

Comparing these two model types on the same data is how the project
satisfies the objective *"compare the proposed model with simple baseline
models."*

---

## Project Structure

```
network-traffic-transformer-lstm/
├── README.md
├── requirements.txt
├── dataset/
│   ├── cicdarknet2020.parquet
│   ├── unsw_nb15.csv
│   └── dataset_link.txt
├── notebooks/
│   └── experiment.ipynb
├── src/
│   ├── preprocessing.py     # load, clean, encode (incl. categorical), scale, split
│   ├── model.py              # transformer_lstm + baseline_lstm architectures
│   ├── train.py               # CLI: train either model on either dataset
│   ├── evaluate.py            # CLI: evaluate a trained model, save metrics+plots
│   ├── compare_results.py     # baseline-vs-proposed AND cross-dataset comparison
│   └── utils.py                # logging, plotting, metrics I/O helpers
├── results/
│   └── <dataset_name>/<model_type>/   # metrics.json, evaluation_report.txt
│       └── comparison/                 # comparison CSVs + text tables
├── figures/
│   └── <dataset_name>/<model_type>/   # training_curves.png, confusion_matrix.png, roc_curve.png
│       └── comparison/                 # comparison bar charts
├── saved_models/
│   └── <dataset_name>/
│       ├── scaler.joblib, label_encoder.joblib, feature_encoders.joblib  (shared)
│       └── <model_type>/best_model.keras, training_history.pkl
└── report/
    └── final_report.pdf   (write this yourself — see note below)
```

---

## Installation

```bash
cd network-traffic-transformer-lstm
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### 1. Train (proposed model + baseline, on each dataset)

```bash
# Proposed model
python src/train.py --dataset_path dataset/cicdarknet2020.parquet --dataset_name cicdarknet2020 --model_type transformer_lstm
python src/train.py --dataset_path dataset/unsw_nb15.csv           --dataset_name unsw_nb15      --model_type transformer_lstm

# Baseline model (for comparison)
python src/train.py --dataset_path dataset/cicdarknet2020.parquet --dataset_name cicdarknet2020 --model_type baseline_lstm
python src/train.py --dataset_path dataset/unsw_nb15.csv           --dataset_name unsw_nb15      --model_type baseline_lstm
```

Optional flags: `--epochs 50 --batch_size 64 --lr 0.001`

### 2. Evaluate (same dataset/model_type combinations)

```bash
python src/evaluate.py --dataset_path dataset/cicdarknet2020.parquet --dataset_name cicdarknet2020 --model_type transformer_lstm
python src/evaluate.py --dataset_path dataset/cicdarknet2020.parquet --dataset_name cicdarknet2020 --model_type baseline_lstm
python src/evaluate.py --dataset_path dataset/unsw_nb15.csv           --dataset_name unsw_nb15      --model_type transformer_lstm
python src/evaluate.py --dataset_path dataset/unsw_nb15.csv           --dataset_name unsw_nb15      --model_type baseline_lstm
```

Each run saves to `results/<dataset_name>/<model_type>/` and
`figures/<dataset_name>/<model_type>/`: accuracy/precision/recall/F1,
confusion matrix, training/validation accuracy+loss curves, and (for binary
datasets) an ROC curve.

### 3. Compare results

```bash
python src/compare_results.py
```

Produces:
- `results/comparison/baseline_vs_proposed.csv` / `.txt` + matching bar
  charts in `figures/comparison/` — **proposed vs. baseline**, per dataset.
- `results/comparison/dataset_comparison.csv` / `.txt` + bar chart —
  **proposed model across both datasets**.

You can run any subset of the 4 train/evaluate combinations above; the
comparison script will use whichever `metrics.json` files it finds and warn
about anything missing.

### 4. Interactive notebook

```bash
jupyter notebook notebooks/experiment.ipynb
```

Edit the `DATASET_PATH` / `DATASET_NAME` / `MODEL_TYPE` config cell and
re-run top to bottom for each of the 4 combinations, then run the final
"Compare" cell.

---

## Preprocessing Details

`src/preprocessing.py` implements the full pipeline required by the project:
- Removes duplicate rows.
- Replaces ±infinity with NaN, then imputes (median for numeric columns,
  mode for categorical columns).
- Auto-detects the target/label column (`Label`, `label`, `Class`,
  `Category`, `Attack`, `Target`, with a last-column fallback).
- Drops pure row-identifier columns (e.g. `id`) and any "sibling" target
  columns (e.g. `attack_cat` when the chosen target is `label`) so the model
  cannot trivially leak the answer.
- **Label-encodes categorical feature columns** (protocol, service, state,
  etc.) instead of discarding them — protocol information is an explicitly
  required feature per the project's Feature Preparation step.
- Standard-scales all numeric features.
- Stratified 70/15/15 train/val/test split.
- Reshapes to `(samples, timesteps=1, features)` for the LSTM/Transformer
  input.
- Persists `scaler.joblib`, `label_encoder.joblib`, and
  `feature_encoders.joblib` so evaluation reuses the exact training-time
  transforms.

---

## Notebook

The notebook mirrors `train.py` / `evaluate.py` exactly (it imports and calls
the same functions), so results match the CLI path. It walks through
preprocessing -> model build -> training -> evaluation -> confusion matrix ->
learning curves -> comparison.

---

## Team Members

| Name | Roll No | Contribution |
|------|---------|---------------|
| _Member 1_ | _____ | Preprocessing, GitHub repo |
| _Member 2_ | _____ | Model design & training |
| _Member 3_ | _____ | Evaluation, report writing |

---

## Note on the Final Report

Per the assignment rules, the IEEE-format report (`report/final_report.pdf`)
must be written by the student team **in their own words**, based on the
**actual numbers** in `results/*/metrics.json` and `results/comparison/`
after running the pipeline on these real datasets — AI-generated report text
is explicitly disallowed and will be graded as zero if detected.

## References

1. Network traffic classification: Techniques, datasets, and challenges.
2. Towards the Deployment of Machine Learning Solutions in Network Traffic
   Classification: A Systematic Survey.
3. Efficient Dark Web traffic classification using a hybrid CNN-LSTM model.
4. Robust Network Traffic Classification.
