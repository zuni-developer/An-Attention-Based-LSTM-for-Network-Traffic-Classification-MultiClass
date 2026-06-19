"""
train.py - Training script for the Transformer-Enhanced LSTM model.

CLI Usage:
    python src/train.py --dataset_path dataset/cicdarknet2020.csv --dataset_name cicdarknet2020
    python src/train.py --dataset_path dataset/network_traffic.csv  --dataset_name network_traffic
"""

import os
import sys
import pickle
import argparse
import logging
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

# Allow running from project root or from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import preprocess
from model import build_model_by_type
from utils import setup_logging, ensure_dir, plot_training_curves, save_plot

logger = setup_logging("train")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    dataset_path: str,
    dataset_name: str,
    model_type: str = "transformer_lstm",
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> dict:
    """
    Full training pipeline.

    Parameters
    ----------
    dataset_path : str
        Path to the raw dataset file (CSV or Parquet).
    dataset_name : str
        Short name used for saving artefacts (e.g. 'cicdarknet2020').
    model_type : str
        Either 'transformer_lstm' (proposed model) or 'baseline_lstm'
        (plain-LSTM baseline used for comparison).
    epochs : int
        Maximum number of training epochs.
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Initial Adam learning rate.

    Returns
    -------
    dict
        Preprocessing data dict (X_train, y_train, etc.) merged with
        'history' key containing the Keras training history dict.
    """
    # ---- Directories ----
    # Preprocessing artefacts (scaler/label_encoder) are shared across model
    # types for the same dataset; model checkpoints are kept per model_type
    # so the baseline and the proposed model don't overwrite each other.
    artifacts_dir = str(ensure_dir(os.path.join("saved_models", dataset_name)))
    model_dir = ensure_dir(os.path.join("saved_models", dataset_name, model_type))
    figures_dir = ensure_dir(os.path.join("figures", dataset_name, model_type))

    # ---- Preprocessing ----
    logger.info("=== Preprocessing (%s) ===", dataset_name)
    data = preprocess(dataset_path, artifacts_dir)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val   = data["X_val"]
    y_val   = data["y_val"]
    num_classes = data["num_classes"]
    is_binary   = data["is_binary"]

    timesteps   = X_train.shape[1]
    num_features = X_train.shape[2]

    # ---- Build Model ----
    logger.info("=== Building Model (%s) ===", model_type)
    model = build_model_by_type(
        model_type=model_type,
        timesteps=timesteps,
        num_features=num_features,
        num_classes=num_classes,
        is_binary=is_binary,
        learning_rate=learning_rate,
    )

    # ---- Callbacks ----
    best_model_path = str(model_dir / "best_model.keras")
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ---- Train ----
    logger.info("=== Training (%s / %s) | epochs=%d | batch=%d ===", dataset_name, model_type, epochs, batch_size)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # ---- Save History ----
    history_path = str(model_dir / "training_history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)
    logger.info("Training history saved → %s", history_path)

    # ---- Plot Training Curves ----
    curve_path = str(figures_dir / "training_curves.png")
    plot_training_curves(
        history.history,
        title=f"Training History – {dataset_name} ({model_type})",
        save_path=curve_path,
    )
    logger.info("Training curves saved → %s", curve_path)

    logger.info("=== Training complete (%s / %s) ===", dataset_name, model_type)

    data["history"] = history.history
    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train Transformer-LSTM on a network traffic dataset."
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to the raw CSV dataset file.",
    )
    parser.add_argument(
        "--dataset_name",
        required=True,
        help="Short dataset name used for artefact sub-directories (e.g. 'cicdarknet2020').",
    )
    parser.add_argument(
        "--model_type",
        choices=["transformer_lstm", "baseline_lstm"],
        default="transformer_lstm",
        help="Which model to train: the proposed Transformer-LSTM or the plain-LSTM baseline (default transformer_lstm).",
    )
    parser.add_argument("--epochs",     type=int,   default=50,   help="Training epochs (default 50).")
    parser.add_argument("--batch_size", type=int,   default=64,   help="Batch size (default 64).")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning rate (default 1e-3).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
