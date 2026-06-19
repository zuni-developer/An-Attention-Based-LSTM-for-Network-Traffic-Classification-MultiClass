"""
preprocessing.py - Robust preprocessing pipeline for network traffic CSV datasets.

Steps:
  1. Load CSV
  2. Remove duplicates
  3. Handle missing / infinite values
  4. Auto-detect target column
  5. Label-encode target
  6. Standard-scale features
  7. 70/15/15 stratified split
  8. Reshape to LSTM format (samples, 1, features)
  9. Persist label_encoder.joblib and scaler.joblib
"""

import os
import logging
import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils import setup_logging, ensure_dir

logger = setup_logging("preprocessing")

# Candidate column names for the target variable (ordered by priority)
_TARGET_CANDIDATES: List[str] = [
    "Label", "label",
    "Class", "class",
    "Category", "category",
    "Attack", "attack",
    "Target", "target",
]

# Columns that must NEVER be used as model features even though they may be
# numeric or look informative: pure row identifiers (no predictive value and
# can leak ordering), and "sibling" target columns that encode the same
# information as the chosen label (e.g. 'attack_cat' alongside 'label' in
# UNSW-NB15) which would otherwise leak the answer directly into the model.
_ID_LIKE_COLUMNS: List[str] = ["id", "Id", "ID", "index", "Unnamed: 0", "flow_id", "Flow ID"]
_SIBLING_TARGET_COLUMNS: List[str] = [
    "Label", "label", "Class", "class", "Category", "category",
    "Attack", "attack", "Target", "target", "attack_cat", "Attack_cat",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file with robust encoding / separator detection."""
    logger.info("Loading CSV: %s", path)
    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
        logger.info("Loaded %d rows × %d columns.", len(df), len(df.columns))
        return df
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed – retrying with latin-1 encoding.")
        df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    logger.info("Loaded %d rows × %d columns.", len(df), len(df.columns))
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw dataframe:
      - Strip whitespace from column names
      - Remove duplicate rows
      - Replace ±inf with NaN
      - Drop rows with all-NaN feature values
      - Fill remaining NaN with column median (numeric) or mode (categorical)
    """
    # Strip column name whitespace
    df.columns = df.columns.str.strip()

    # Deduplicate
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Removed %d duplicate rows.", before - len(df))

    # Replace infinities
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows that are entirely NaN
    before = len(df)
    df = df.dropna(how="all")
    logger.info("Dropped %d all-NaN rows.", before - len(df))

    # Fill remaining NaN per column
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "UNKNOWN")

    logger.info("Missing values after imputation: %d", df.isna().sum().sum())
    return df.reset_index(drop=True)


def detect_target_column(df: pd.DataFrame) -> str:
    """
    Detect the target column by checking candidate names against df columns.
    Falls back to the last column if none of the candidates are found.
    """
    for candidate in _TARGET_CANDIDATES:
        if candidate in df.columns:
            logger.info("Target column detected: '%s'", candidate)
            return candidate

    # Fallback: last column
    target = df.columns[-1]
    logger.warning(
        "No standard target column found. Using last column: '%s'", target
    )
    return target


def encode_labels(series: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    """Label-encode a target series. Returns encoded array and fitted encoder."""
    le = LabelEncoder()
    encoded = le.fit_transform(series.astype(str))
    logger.info(
        "Classes (%d): %s", len(le.classes_), list(le.classes_[:10])
    )
    return encoded, le


def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit a StandardScaler on X_train and transform all splits."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def reshape_for_lstm(X: np.ndarray, timesteps: int = 1) -> np.ndarray:
    """Reshape 2-D feature array to (samples, timesteps, features) for LSTM."""
    return X.reshape((X.shape[0], timesteps, X.shape[1]))


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.70,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified 70/15/15 split.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    test_size = 1.0 - train_size - val_size  # 0.15

    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(1.0 - train_size),
            random_state=random_state,
            stratify=y,
        )
        relative_val = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1.0 - relative_val),
            random_state=random_state,
            stratify=y_temp,
        )
    except ValueError:
        # Fallback: no stratification if classes too rare
        logger.warning("Stratified split failed – falling back to random split.")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(1.0 - train_size), random_state=random_state
        )
        relative_val = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1.0 - relative_val), random_state=random_state
        )

    logger.info(
        "Split → Train: %d | Val: %d | Test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess(
    csv_path: str,
    artifacts_dir: str,
    timesteps: int = 1,
) -> dict:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    csv_path : str
        Path to raw CSV file.
    artifacts_dir : str
        Directory to save scaler / label_encoder artefacts.
    timesteps : int
        Number of LSTM timesteps (default 1).

    Returns
    -------
    dict with keys:
        X_train, X_val, X_test  – shaped (N, timesteps, features)
        y_train, y_val, y_test  – 1-D integer arrays
        num_classes             – int
        is_binary               – bool
        label_encoder           – fitted LabelEncoder
        scaler                  – fitted StandardScaler
        feature_names           – list of feature column names
    """
    ensure_dir(artifacts_dir)

    df = load_csv(csv_path)
    df = clean_dataframe(df)

    target_col = detect_target_column(df)

    y_raw = df[target_col]
    X_df = df.drop(columns=[target_col])

    # Drop pure row-identifier columns (no predictive value, e.g. 'id')
    id_cols_present = [c for c in X_df.columns if c in _ID_LIKE_COLUMNS]
    if id_cols_present:
        logger.info("Dropping ID-like columns: %s", id_cols_present)
        X_df = X_df.drop(columns=id_cols_present)

    # Drop sibling target columns (e.g. 'attack_cat' when target is 'label')
    # to prevent the model from directly leaking the answer.
    sibling_cols_present = [
        c for c in X_df.columns if c in _SIBLING_TARGET_COLUMNS and c != target_col
    ]
    if sibling_cols_present:
        logger.info("Dropping sibling target columns to avoid leakage: %s", sibling_cols_present)
        X_df = X_df.drop(columns=sibling_cols_present)

    if X_df.empty or X_df.shape[1] == 0:
        raise ValueError("No feature columns remain after dropping target/ID/leak columns.")

    # Encode categorical (non-numeric) feature columns instead of dropping
    # them — protocol/service/state-type fields are explicitly required
    # traffic-behavior features for this project, not noise.
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]

    feature_encoders = {}
    for col in categorical_cols:
        le_col = LabelEncoder()
        X_df[col] = le_col.fit_transform(X_df[col].astype(str))
        feature_encoders[col] = le_col

    if categorical_cols:
        logger.info("Label-encoded categorical feature columns: %s", categorical_cols)

    feature_names = X_df.columns.tolist()
    logger.info("Number of features: %d", len(feature_names))

    X = X_df.values.astype(np.float32)
    y, label_encoder = encode_labels(y_raw)

    num_classes = len(label_encoder.classes_)
    is_binary = num_classes == 2
    logger.info("num_classes=%d  is_binary=%s", num_classes, is_binary)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)

    # Reshape → LSTM
    X_train = reshape_for_lstm(X_train, timesteps)
    X_val = reshape_for_lstm(X_val, timesteps)
    X_test = reshape_for_lstm(X_test, timesteps)

    # Persist artefacts
    le_path = os.path.join(artifacts_dir, "label_encoder.joblib")
    sc_path = os.path.join(artifacts_dir, "scaler.joblib")
    fe_path = os.path.join(artifacts_dir, "feature_encoders.joblib")
    joblib.dump(label_encoder, le_path)
    joblib.dump(scaler, sc_path)
    joblib.dump(feature_encoders, fe_path)
    logger.info("Saved label_encoder → %s", le_path)
    logger.info("Saved scaler       → %s", sc_path)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "num_classes": num_classes,
        "is_binary": is_binary,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "feature_names": feature_names,
        "feature_encoders": feature_encoders,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Preprocess a network traffic CSV dataset.")
    parser.add_argument("--dataset_path", required=True, help="Path to raw CSV file.")
    parser.add_argument(
        "--artifacts_dir",
        default="saved_models",
        help="Directory to save scaler / label_encoder artefacts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = preprocess(args.dataset_path, args.artifacts_dir)
    print("\nPreprocessing complete.")
    print(f"  X_train shape : {result['X_train'].shape}")
    print(f"  X_val   shape : {result['X_val'].shape}")
    print(f"  X_test  shape : {result['X_test'].shape}")
    print(f"  num_classes   : {result['num_classes']}")
    print(f"  is_binary     : {result['is_binary']}")
