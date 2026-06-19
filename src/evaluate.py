"""
evaluate.py - Evaluation script for trained Transformer-LSTM models.

CLI Usage:
    python src/evaluate.py --dataset_path dataset/cicdarknet2020.csv --dataset_name cicdarknet2020
    python src/evaluate.py --dataset_path dataset/network_traffic.csv  --dataset_name network_traffic
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

# Allow running from project root or from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import preprocess
from utils import (
    setup_logging,
    ensure_dir,
    plot_confusion_matrix,
    plot_roc_curve,
    save_metrics_json,
    save_text_report,
)

logger = setup_logging("evaluate")


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(dataset_path: str, dataset_name: str, model_type: str = "transformer_lstm") -> dict:
    """
    Load a trained model and evaluate it on the held-out test split.

    Parameters
    ----------
    dataset_path : str
        Path to the raw dataset file (same file used during training).
    dataset_name : str
        Short name matching the artefact sub-directories.
    model_type : str
        Either 'transformer_lstm' (proposed model) or 'baseline_lstm'.

    Returns
    -------
    dict
        Evaluation metrics dict (accuracy, precision, recall, f1, optionally roc_auc).
    """
    artifacts_dir = Path("saved_models") / dataset_name
    model_dir   = artifacts_dir / model_type
    results_dir = ensure_dir(Path("results") / dataset_name / model_type)
    figures_dir = ensure_dir(Path("figures") / dataset_name / model_type)

    # ---- Load artefacts ----
    model_path = model_dir / "best_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            f"Please run: python src/train.py --dataset_path {dataset_path} "
            f"--dataset_name {dataset_name} --model_type {model_type}"
        )

    logger.info("Loading model from %s", model_path)
    model = tf.keras.models.load_model(str(model_path))

    le_path = artifacts_dir / "label_encoder.joblib"
    sc_path = artifacts_dir / "scaler.joblib"
    label_encoder = joblib.load(str(le_path))
    scaler        = joblib.load(str(sc_path))

    # ---- Preprocess (re-uses saved artefacts automatically) ----
    logger.info("Preprocessing dataset for evaluation …")
    data = preprocess(dataset_path, str(artifacts_dir))

    X_test  = data["X_test"]
    y_test  = data["y_test"]
    is_binary = data["is_binary"]
    num_classes = data["num_classes"]
    class_names = list(label_encoder.classes_)

    # ---- Predict ----
    logger.info("Running inference on %d test samples …", len(X_test))
    y_prob = model.predict(X_test, verbose=0)

    if is_binary:
        y_prob_flat = y_prob.ravel()
        y_pred = (y_prob_flat >= 0.5).astype(int)
    else:
        y_pred = np.argmax(y_prob, axis=1)

    # ---- Metrics ----
    avg = "binary" if is_binary else "weighted"

    accuracy  = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average=avg, zero_division=0))
    recall    = float(recall_score(y_test, y_pred, average=avg, zero_division=0))
    f1        = float(f1_score(y_test, y_pred, average=avg, zero_division=0))
    cm        = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))
    report    = classification_report(
        y_test, y_pred,
        labels=list(range(num_classes)),
        target_names=[str(c) for c in class_names],
        zero_division=0,
    )
    metrics = {
        "dataset_name": dataset_name,
        "model_type":   model_type,
        "accuracy":     accuracy,
        "precision":    precision,
        "recall":       recall,
        "f1_score":     f1,
        "confusion_matrix": cm.tolist(),
    }

    logger.info("Accuracy  : %.4f", accuracy)
    logger.info("Precision : %.4f", precision)
    logger.info("Recall    : %.4f", recall)
    logger.info("F1-Score  : %.4f", f1)

    # ---- ROC-AUC (binary only) ----
    if is_binary:
        try:
            roc_auc = float(roc_auc_score(y_test, y_prob_flat))
            fpr, tpr, _ = roc_curve(y_test, y_prob_flat)
            metrics["roc_auc"] = roc_auc
            logger.info("ROC-AUC   : %.4f", roc_auc)

            roc_path = str(figures_dir / "roc_curve.png")
            plot_roc_curve(
                fpr, tpr, roc_auc,
                title=f"ROC Curve – {dataset_name} ({model_type})",
                save_path=roc_path,
            )
            logger.info("ROC curve saved → %s", roc_path)
        except Exception as exc:
            logger.warning("Could not compute ROC-AUC: %s", exc)

    # ---- Confusion Matrix Plot ----
    cm_path = str(figures_dir / "confusion_matrix.png")
    plot_confusion_matrix(
        cm,
        class_names=class_names,
        title=f"Confusion Matrix – {dataset_name} ({model_type})",
        save_path=cm_path,
    )
    logger.info("Confusion matrix saved → %s", cm_path)

    # ---- Save Results ----
    json_path = str(results_dir / "metrics.json")
    save_metrics_json(metrics, json_path)
    logger.info("Metrics JSON saved → %s", json_path)

    txt_report = (
        f"=== Evaluation Report: {dataset_name} ({model_type}) ===\n\n"
        f"Accuracy  : {accuracy:.4f}\n"
        f"Precision : {precision:.4f}\n"
        f"Recall    : {recall:.4f}\n"
        f"F1-Score  : {f1:.4f}\n"
    )
    if "roc_auc" in metrics:
        txt_report += f"ROC-AUC   : {metrics['roc_auc']:.4f}\n"
    txt_report += f"\n{report}\n"

    txt_path = str(results_dir / "evaluation_report.txt")
    save_text_report(txt_report, txt_path)
    logger.info("Text report saved → %s", txt_path)

    logger.info("=== Evaluation complete (%s / %s) ===", dataset_name, model_type)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Transformer-LSTM model on network traffic data."
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to the raw CSV dataset file.",
    )
    parser.add_argument(
        "--dataset_name",
        required=True,
        help="Short dataset name (e.g. 'cicdarknet2020').",
    )
    parser.add_argument(
        "--model_type",
        choices=["transformer_lstm", "baseline_lstm"],
        default="transformer_lstm",
        help="Which trained model to evaluate (default transformer_lstm).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    metrics = evaluate(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        model_type=args.model_type,
    )
    print("\n=== Final Metrics ===")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:15s}: {v}")
