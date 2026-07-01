"""
model.py - Transformer-Enhanced LSTM model using TensorFlow/Keras Functional API.

Architecture:
  Input(shape=(timesteps, features))
  → LSTM(128, return_sequences=True)
  → MultiHeadAttention(num_heads=4, key_dim=32)  [with residual + LayerNorm]
  → GlobalAveragePooling1D
  → Dropout(0.3)
  → Dense(128, relu)
  → Dropout(0.3)
  → Output Dense (sigmoid for binary, softmax for multi-class)
"""

import logging
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    LSTM,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D,
    Dropout,
    Dense,
    Add,
)
from tensorflow.keras.optimizers import Adam

from utils import setup_logging

logger = setup_logging("model")


def build_model(
    timesteps: int,
    num_features: int,
    num_classes: int,
    is_binary: bool,
    lstm_units: int = 128,
    num_heads: int = 4,
    key_dim: int = 32,
    dense_units: int = 128,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> Model:
    """
    Build and compile the Transformer-Enhanced LSTM model.

    Parameters
    ----------
    timesteps : int
        Number of time steps per sample (1 for tabular data reformatted as sequences).
    num_features : int
        Number of input features per time step.
    num_classes : int
        Total number of target classes.
    is_binary : bool
        If True uses sigmoid output + binary_crossentropy;
        otherwise uses softmax + sparse_categorical_crossentropy.
    lstm_units : int
        Number of hidden units in the LSTM layer.
    num_heads : int
        Number of attention heads in MultiHeadAttention.
    key_dim : int
        Dimensionality of each attention head.
    dense_units : int
        Number of units in the intermediate Dense layer.
    dropout_rate : float
        Dropout probability applied after attention and after the dense block.
    learning_rate : float
        Initial learning rate for the Adam optimiser.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model.
    """
    inputs = Input(shape=(timesteps, num_features), name="input")

    # ---- LSTM ----
    lstm_out = LSTM(lstm_units, return_sequences=True, name="lstm")(inputs)

    # ---- Multi-Head Self-Attention ----
    attn_out = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        name="multi_head_attention",
    )(lstm_out, lstm_out)

    # ---- Residual Connection + Layer Normalisation ----
    # Project lstm_out to attention dimension if shapes differ
    if lstm_out.shape[-1] != attn_out.shape[-1]:
        lstm_proj = Dense(attn_out.shape[-1], use_bias=False, name="lstm_proj")(lstm_out)
        residual = Add(name="residual_add")([lstm_proj, attn_out])
    else:
        residual = Add(name="residual_add")([lstm_out, attn_out])

    norm_out = LayerNormalization(name="layer_norm")(residual)

    # ---- Pooling + Classification Head ----
    pooled = GlobalAveragePooling1D(name="global_avg_pool")(norm_out)
    x = Dropout(dropout_rate, name="dropout_1")(pooled)
    x = Dense(dense_units, activation="relu", name="dense_1")(x)
    x = Dropout(dropout_rate, name="dropout_2")(x)

    # ---- Output Layer ----
    if is_binary:
        outputs = Dense(1, activation="sigmoid", name="output")(x)
        loss = "binary_crossentropy"
    else:
        outputs = Dense(num_classes, activation="softmax", name="output")(x)
        loss = "sparse_categorical_crossentropy"

    model = Model(inputs=inputs, outputs=outputs, name="TransformerLSTM")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )

    logger.info("Model built successfully.")
    logger.info("  Input shape  : (%s, %d, %d)", "None", timesteps, num_features)
    logger.info("  Output units : %d  |  loss: %s", 1 if is_binary else num_classes, loss)
    model.summary(print_fn=logger.info)

    return model



def build_model_by_type(
    model_type: str,
    timesteps: int,
    num_features: int,
    num_classes: int,
    is_binary: bool,
    learning_rate: float = 1e-3,
) -> Model:
    """
    Factory function: build either the proposed Transformer-Enhanced LSTM
    ('transformer_lstm') or the plain-LSTM baseline ('baseline_lstm').
    """
    if model_type == "transformer_lstm":
        return build_model(
            timesteps=timesteps,
            num_features=num_features,
            num_classes=num_classes,
            is_binary=is_binary,
            learning_rate=learning_rate,
        )
    # elif model_type == "baseline_lstm":
    #     return build_baseline_model(
    #         timesteps=timesteps,
    #         num_features=num_features,
    #         num_classes=num_classes,
    #         is_binary=is_binary,
    #         learning_rate=learning_rate,
    #     )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Expected 'transformer_lstm' or 'baseline_lstm'."
        )


def get_model_summary(model: Model) -> str:
    """Return model.summary() as a string."""
    lines = []
    model.summary(print_fn=lambda line: lines.append(line))
    return "\n".join(lines)
