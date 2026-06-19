"""
utils.py - Utility functions for Network Traffic Classification project.
Handles plot saving, confusion matrix visualization, metrics JSON export,
and directory creation.
"""

import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def setup_logging(name: str = "network_traffic", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def ensure_dir(path: str) -> Path:
    """Create directory (and parents) if it does not exist. Return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_plot(fig: plt.Figure, save_path: str, dpi: int = 150) -> None:
    """Save a matplotlib figure to disk and close it."""
    ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str = "Confusion Matrix",
    save_path: str = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Plot and optionally save a confusion matrix as a heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix array (n_classes x n_classes).
    class_names : list
        List of class label strings.
    title : str
        Plot title.
    save_path : str, optional
        File path to save the figure.
    figsize : tuple
        Figure dimensions in inches.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Normalise for percentages
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig


def plot_training_curves(
    history: dict,
    title: str = "Training History",
    save_path: str = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Plot training & validation accuracy and loss curves.

    Parameters
    ----------
    history : dict
        Keras history.history dict (or equivalent loaded dict).
    title : str
        Overall figure title.
    save_path : str, optional
        File path to save the figure.
    figsize : tuple
        Figure dimensions.

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # --- Accuracy ---
    axes[0].plot(history.get("accuracy", []), label="Train Accuracy", linewidth=2)
    axes[0].plot(history.get("val_accuracy", []), label="Val Accuracy", linewidth=2, linestyle="--")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Loss ---
    axes[1].plot(history.get("loss", []), label="Train Loss", linewidth=2)
    axes[1].plot(history.get("val_loss", []), label="Val Loss", linewidth=2, linestyle="--")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig


def save_metrics_json(metrics: dict, save_path: str) -> None:
    """Serialise a metrics dictionary to a JSON file."""
    ensure_dir(os.path.dirname(save_path))
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, default=_json_serializer)


def load_metrics_json(path: str) -> dict:
    """Load a metrics JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text_report(text: str, save_path: str) -> None:
    """Write a plain-text report to disk."""
    ensure_dir(os.path.dirname(save_path))
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)


def _json_serializer(obj):
    """Custom JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    title: str = "ROC Curve",
    save_path: str = None,
    figsize: tuple = (7, 6),
) -> plt.Figure:
    """
    Plot an ROC curve (binary classification only).

    Parameters
    ----------
    fpr : np.ndarray
        False positive rates.
    tpr : np.ndarray
        True positive rates.
    roc_auc : float
        Area under the ROC curve.
    title : str
        Plot title.
    save_path : str, optional
        File path to save the figure.
    figsize : tuple
        Figure dimensions.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig
