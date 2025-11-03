#!/usr/bin/env python
"""
Transformer Model Training

Architecture: Embedding → Positional Encoding → Multi-Head Attention → FFN → Dense → Sigmoid

"""

import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Dropout, LayerNormalization,
    GlobalAveragePooling1D, MultiHeadAttention, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

from data_utils import load_and_prepare_data, save_mlinfo


def positional_encoding(max_length, d_model):
    """
    Create positional encoding matrix

    Args:
        max_length: Maximum sequence length
        d_model: Model dimension (embedding size)

    Returns:
        Positional encoding matrix of shape (1, max_length, d_model)
    """
    position = np.arange(max_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pos_encoding = np.zeros((max_length, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)

    return pos_encoding[np.newaxis, ...]


class TransformerBlock(tf.keras.layers.Layer):
    """Single Transformer block with multi-head attention and feed-forward network"""

    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(d_model),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training=False):
        # Multi-head attention
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout_rate
        })
        return config


def build_transformer_model(vocab_size, max_length=128, d_model=128, num_heads=4,
                            ff_dim=256, num_blocks=2, dropout=0.1):
    """
    Build Transformer model for sequence classification

    Args:
        vocab_size: Size of character vocabulary
        max_length: Maximum sequence length
        d_model: Model dimension (embedding size)
        num_heads: Number of attention heads
        ff_dim: Feed-forward network dimension
        num_blocks: Number of transformer blocks
        dropout: Dropout rate
    """
    # Input layer
    inputs = Input(shape=(max_length,), name='input')

    # Embedding layer
    x = Embedding(
        vocab_size,
        d_model,
        embeddings_initializer='normal',
        mask_zero=True
    )(inputs)

    # Add positional encoding
    pos_enc = positional_encoding(max_length, d_model)
    x = x + pos_enc

    # Transformer blocks
    for i in range(num_blocks):
        x = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )(x)

    # Global average pooling
    x = GlobalAveragePooling1D()(x)

    # Dropout before final layer
    x = Dropout(dropout)(x)

    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='Transformer')

    # Compile
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    return model


def train(args):
    """Main training pipeline"""

    print("="*70)
    print("TRANSFORMER MODEL TRAINING")
    print("="*70)
    print()

    # Determine output directory
    import os
    output_dir = os.environ.get('OUTPUT_DIR', 'output/transformer')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Load and prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, char_list, vocab_size = \
        load_and_prepare_data(max_length=args.max_length)

    # Build model
    print("\nBuilding Transformer model...")
    model = build_transformer_model(
        vocab_size=vocab_size,
        max_length=args.max_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout
    )

    model.summary()

    # Training callbacks
    model_file = os.path.join(output_dir, f'transformer_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.hdf5')

    callbacks = [
        EarlyStopping(monitor='val_loss',
            patience=args.patience,
            verbose=1,
            restore_best_weights=True
        ),
        ModelCheckpoint(monitor='val_loss',
            filepath=model_file,
            verbose=1,
            save_best_only=True
        ),
        ReduceLROnPlateau(
            patience=3,
            factor=0.5,
            verbose=1,
            min_lr=1e-6
        )
    ]

    # Class weights
    class_weight = {0: 1.0, 1: 1.12}

    # Train
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)

    history = model.fit(
        X_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        shuffle=True,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weight
    )

    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION ON TEST SET")
    print("="*70)

    # Get best model (use trained model if callback didn't save)
    from tensorflow.keras.models import load_model
    try:
        if os.path.exists(model_file):
            best_model = load_model(model_file, custom_objects={'TransformerBlock': TransformerBlock})
            print(f"Loaded checkpoint from: {model_file}")
        else:
            print(f"Checkpoint not found at {model_file}, using current model")
            best_model = model
    except Exception as e:
        print(f"Warning: Could not load checkpoint ({e}), using current model")
        best_model = model

    # Predictions
    y_pred_proba = best_model.predict(X_test, batch_size=64, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Metrics
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=['Legitimate', 'Suspicious'],
        digits=4
    ))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
    precision = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
    recall = cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # ===== CRITICAL: FORCE SAVE MODEL =====
    print(f"\nForce-saving model to guarantee output...")
    try:
        best_model.save(model_file)
        print(f"✓ Model saved to: {model_file}")
        print(f"  File size: {os.path.getsize(model_file) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"⚠️ Error saving model: {e}")
        print(f"  File exists: {os.path.exists(model_file)}")

    # Save mlinfo
    mlinfo_file = os.path.join(output_dir, f'transformer_mlinfo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    mlinfo = save_mlinfo(
        os.path.basename(model_file),
        char_list,
        vocab_size,
        args.max_length,
        history.history,
        output_file=mlinfo_file
    )

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nModel: {model_file}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test F1-Score: {f1:.4f}")

    return model_file, mlinfo


def main():
    parser = argparse.ArgumentParser(description='Train Transformer model')

    # Model hyperparameters
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--ff-dim', type=int, default=256, help='Feed-forward dimension')
    parser.add_argument('--num-blocks', type=int, default=2, help='Number of transformer blocks')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max-length', type=int, default=128, help='Max sequence length')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=1000, help='Maximum epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
