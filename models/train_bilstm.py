#!/usr/bin/env python
"""
Bidirectional LSTM Model Training

Architecture: Embedding → BiLSTM → BiLSTM → Dropout → Dense → Sigmoid
"""

import argparse
from datetime import datetime
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

from data_utils import load_and_prepare_data, save_mlinfo


def build_bilstm_model(vocab_size, max_length=128, embedding_dim=128, lstm_units=128, dropout=0.3):
    """
    Build Bidirectional LSTM model

    Args:
        vocab_size: Size of character vocabulary
        max_length: Maximum sequence length
        embedding_dim: Dimension of character embeddings
        lstm_units: Number of LSTM units
        dropout: Dropout rate
    """

    model = Sequential(name='BiLSTM')

    # Embedding layer
    model.add(Embedding(
        vocab_size,
        embedding_dim,
        embeddings_initializer='normal',
        input_length=max_length,
        mask_zero=True
    ))

    # First BiLSTM layer (returns sequences)
    model.add(Bidirectional(
        LSTM(lstm_units, return_sequences=True),
        name='bilstm_1'
    ))

    # Second BiLSTM layer
    model.add(Bidirectional(
        LSTM(lstm_units),
        name='bilstm_2'
    ))

    # Dropout for regularization
    model.add(Dropout(dropout))

    # Output layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # Compile
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )

    return model


def train(args):
    """Main training pipeline"""

    print("="*70)
    print("BIDIRECTIONAL LSTM MODEL TRAINING")
    print("="*70)
    print()

    # Determine output directory
    import os
    output_dir = os.environ.get('OUTPUT_DIR', 'output/bilstm')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Load and prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, char_list, vocab_size = \
        load_and_prepare_data(max_length=args.max_length)

    # Build model
    print("\nBuilding Bidirectional LSTM model...")
    model = build_bilstm_model(
        vocab_size=vocab_size,
        max_length=args.max_length,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
        dropout=args.dropout
    )

    model.summary()

    # Training callbacks
    model_file = os.path.join(output_dir, f'bilstm_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.hdf5')

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            verbose=1,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=model_file,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
        ),
        ReduceLROnPlateau(
            patience=3,
            factor=0.5,
            verbose=1,
            min_lr=1e-6,
        )
    ]

    # Class weights for balanced training
    class_weight = {0: 1.0, 1: 1.12}  # Balance based on data distribution

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
            best_model = load_model(model_file)
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
    print(f"\nTrue Negatives: {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives: {cm[1][1]}")

    # Calculate additional metrics
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
    mlinfo_file = os.path.join(output_dir, f'bilstm_mlinfo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    mlinfo = save_mlinfo(
        os.path.basename(model_file),  # Store relative filename in mlinfo
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
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test F1-Score: {f1:.4f}")

    return model_file, mlinfo


def main():
    parser = argparse.ArgumentParser(description='Train Bidirectional LSTM model')

    # Model hyperparameters
    parser.add_argument('--embedding-dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--lstm-units', type=int, default=128, help='LSTM units')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--max-length', type=int, default=128, help='Max sequence length')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=1000, help='Maximum epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
