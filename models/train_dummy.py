#!/usr/bin/env python
"""
Android Package Name Classifier

Usage:
    python train.py                    # Train with default settings
    python train.py --epochs 100       # Custom epochs
    python train.py --batch-size 512   # Custom batch size
"""

import csv
import codecs
import json
import os
import argparse
from datetime import datetime

import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report


def load_data(label_0_files, label_1_files):
    """Load and combine data from CSV files"""
    print("Loading data...")
    data = []

    for filename in label_0_files:
        if os.path.exists(filename):
            with codecs.open(filename, 'r', 'utf8') as f:
                reader = csv.reader(f)
                for line in reader:
                    data.append((0, line[0]))  # (label, package_name)

    for filename in label_1_files:
        if os.path.exists(filename):
            with codecs.open(filename, 'r', 'utf8') as f:
                reader = csv.reader(f)
                for line in reader:
                    data.append((1, line[0]))

    return data


def prepare_vocabulary(training_data):
    """Create sorted vocabulary from training data"""
    print("Building vocabulary...")
    all_chars = ''.join([pkg for _, pkg in training_data])
    char_list = sorted(list(set(all_chars)))
    char_indices = dict((c, i+1) for i, c in enumerate(char_list))

    print(f"Vocabulary size: {len(char_list)} characters")
    return char_list, char_indices


def encode_data(data, char_indices, max_length=128):
    """Convert package names to padded sequences"""
    X, y = [], []

    for label, package_name in data:
        # Convert characters to indices
        indices = []
        for char in package_name:
            if char in char_indices:
                indices.append(char_indices[char])
            else:
                indices.append(0)  # Unknown character

        # Pad sequence
        padded = sequence.pad_sequences([indices], maxlen=max_length)[0]
        X.append(padded)
        y.append(label)

    return np.array(X).astype(np.uint8), np.array(y)


def build_model(vocab_size, max_length=128):
    """Build LSTM model"""
    model = Sequential()
    model.add(Embedding(vocab_size, 128, embeddings_initializer='normal',
                       input_length=max_length, mask_zero=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=["accuracy"])
    return model


def train(args):
    """Main training pipeline"""

    # Load data (files are in /workspace/data when running in Docker)
    data_dir = '/workspace/data' if os.path.exists('/workspace/data') else '.'
    # Determine output directory
    output_dir = os.environ.get('OUTPUT_DIR', 'output/dummy')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    train_data = load_data([f'{data_dir}/0_train.csv'], [f'{data_dir}/1_train.csv'])
    val_data = load_data([f'{data_dir}/0_validation.csv'], [f'{data_dir}/1_validation.csv'])
    test_data = load_data([f'{data_dir}/0_test.csv'], [f'{data_dir}/1_test.csv'])

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    # Build vocabulary from training data only
    char_list, char_indices = prepare_vocabulary(train_data)

    # Encode datasets
    print("\nEncoding datasets...")
    X_train, y_train = encode_data(train_data, char_indices, args.max_length)
    X_val, y_val = encode_data(val_data, char_indices, args.max_length)
    X_test, y_test = encode_data(test_data, char_indices, args.max_length)

    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # Build model
    print("\nBuilding model...")
    vocab_size = len(char_list) + 1  # +1 for index 0 (unknown/padding)
    model = build_model(vocab_size, args.max_length)
    model.summary()

    # Training callbacks
    model_file = os.path.join(output_dir, f'dummy_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.hdf5')
    print(f"\n📁 Model will be saved to: {model_file}")
    print(f"✓ Output directory exists: {os.path.isdir(output_dir)}")

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=args.patience, verbose=1),
        ModelCheckpoint(
            filepath=model_file,
            monitor='val_loss',
            verbose=1,
            save_best_only=True
        )
    ]

    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        shuffle=True,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET")
    print("="*60)

    # Load best model
    from tensorflow.keras.models import load_model

    # Get best model (use trained model if callback didn't save)
    try:
        if os.path.exists(model_file):
            best_model = load_model(model_file)
            print(f"✓ Loaded checkpoint from: {model_file}")
        else:
            print(f"\n⚠️ Checkpoint not found at {model_file}, using current model")
            best_model = model
    except Exception as e:
        print(f"\n⚠️ Warning: Could not load checkpoint ({e}), using current model")
        best_model = model

    # Predictions
    y_pred_proba = best_model.predict(X_test, batch_size=64, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=['Legitimate', 'Suspicious'],
        digits=4
    ))

    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nTrue Negatives: {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives: {cm[1][1]}")

    # Calculate metrics
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

    # Save mlinfo.json with embedded vocabulary
    version = int(datetime.now().strftime("%Y%m%d%H"))
    mlinfo = {
        'version': version,
        'model': os.path.basename(model_file),
        'vocabulary': {
            'char_list': char_list,
            'max_length': args.max_length,
            'vocab_size': vocab_size
        },
        'training': {
            'epochs_run': len(history.history['loss']),
            'final_train_acc': float(history.history['accuracy'][-1]),
            'final_val_acc': float(history.history['val_accuracy'][-1]),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'best_val_loss': float(min(history.history['val_loss']))
        },
        'test_metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        },
        'created_at': datetime.now().isoformat()
    }

    mlinfo_file = os.path.join(output_dir, f'dummy_mlinfo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(mlinfo_file, 'w') as f:
        json.dump(mlinfo, f, indent=2)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"✓ Model: {model_file}")
    print(f"  Exists: {os.path.exists(model_file)}")
    print(f"✓ Config: {mlinfo_file}")
    print(f"  Exists: {os.path.exists(mlinfo_file)}")
    print(f"✓ Version: {version}")

    return model_file, mlinfo_file


def main():
    parser = argparse.ArgumentParser(description='Train package name classifier')
    parser.add_argument('--epochs', type=int, default=1000, help='Maximum epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience')
    parser.add_argument('--max-length', type=int, default=128, help='Max sequence length')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
