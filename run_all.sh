#!/usr/bin/env bash
# Convenience script: trains + evaluates BOTH model types (proposed
# Transformer-LSTM and the plain-LSTM baseline) on BOTH datasets, then
# builds the comparison tables/plots. Equivalent to running the 8 commands
# in the README by hand.
set -e

DATASETS=("dataset/cicdarknet2020.parquet:cicdarknet2020" "dataset/unsw_nb15.csv:unsw_nb15")
MODELS=("transformer_lstm" "baseline_lstm")

for entry in "${DATASETS[@]}"; do
  IFS=":" read -r path name <<< "$entry"
  for model in "${MODELS[@]}"; do
    echo ""
    echo "=== Training $model on $name ==="
    python src/train.py --dataset_path "$path" --dataset_name "$name" --model_type "$model"
    echo "=== Evaluating $model on $name ==="
    python src/evaluate.py --dataset_path "$path" --dataset_name "$name" --model_type "$model"
  done
done

echo ""
echo "=== Building comparison tables/plots ==="
python src/compare_results.py
