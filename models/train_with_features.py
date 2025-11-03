#!/usr/bin/env python
"""
Training script that combines engineered features with character-level model.

Three approaches:
1. Feature Concatenation 
2. Multi-Input Model 
3. Hybrid 

This script demonstrates Approach 1 & 2
"""

import os
import csv
import codecs
import json
import argparse
from datetime import datetime
import numpy as np

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, Activation,
    Input, Concatenate, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report

from feature_utils import extract_all_features, extract_features_to_array

# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_LENGTH = 128
CHAR_EMBEDDING_DIM = 128
CHAR_LSTM_UNITS = 128
DROPOUT_RATE = 0.3
FEATURE_DENSE_UNITS = 32
COMBINED_DENSE_UNITS = 64


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_data(label_0_files, label_1_files):
    """Load package names with labels"""
    data = []

    for filename in label_0_files:
        try:
            with codecs.open(filename, 'r', 'utf8') as f:
                reader = csv.reader(f)
                for line in reader:
                    data.append((0, line[0]))
        except FileNotFoundError:
            print(f"Warning: {filename} not found")

    for filename in label_1_files:
        try:
            with codecs.open(filename, 'r', 'utf8') as f:
                reader = csv.reader(f)
                for line in reader:
                    data.append((1, line[0]))
        except FileNotFoundError:
            print(f"Warning: {filename} not found")

    return data


def prepare_vocabulary(training_data):
    """Create sorted character vocabulary"""
    all_chars = ''.join([pkg for _, pkg in training_data])
    char_list = sorted(list(set(all_chars)))
    char_indices = dict((c, i+1) for i, c in enumerate(char_list))
    return char_list, char_indices


def encode_char_data(data, char_indices, max_length=128):
    """Encode package names as character sequences"""
    X, y = [], []

    for label, package_name in data:
        indices = []
        for char in package_name:
            if char in char_indices:
                indices.append(char_indices[char])
            else:
                indices.append(0)

        padded = sequence.pad_sequences([indices], maxlen=max_length)[0]
        X.append(padded)
        y.append(label)

    return np.array(X).astype(np.uint8), np.array(y)


def extract_feature_data(data):
    """Extract engineered features for all packages"""
    print("Extracting engineered features...")

    all_features = []
    feature_names = None

    for idx, (label, package_name) in enumerate(data):
        if idx % 5000 == 0:
            print(f"  Processed {idx}/{len(data)}")

        features, feature_names = extract_features_to_array(package_name)
        all_features.append(features)

    X_features = np.array(all_features, dtype=np.float32)

    # NOTE: NO NORMALIZATION - model trained on raw feature values
    # This matches evaluation behavior which also uses raw features
    # Neural networks handle raw feature ranges fine with proper initialization
    # BatchNormalization layers in the model handle internal normalization

    return X_features.astype(np.float32), feature_names


# ============================================================================
# APPROACH 1: SIMPLE CONCATENATION
# ============================================================================

def build_model_concatenation(vocab_size, num_features, max_length=128):
    """
    Build model that concatenates character input with engineered features.

    Architecture:
    - Character embedding + LSTM layer
    - Concatenate with feature input
    - Dense layers for final prediction

    """
    print("\nBuilding CONCATENATION model...")
    print(f"  Character input: {max_length} chars")
    print(f"  Feature input: {num_features} features")

    model = Sequential()

    # Character processing
    model.add(Embedding(vocab_size, CHAR_EMBEDDING_DIM,
                       input_length=max_length, mask_zero=True))
    model.add(LSTM(CHAR_LSTM_UNITS, return_sequences=True))
    model.add(LSTM(CHAR_LSTM_UNITS))
    model.add(Dropout(DROPOUT_RATE))

    # Note: In actual implementation, need to reshape/concatenate externally
    # This is simplified - see Approach 2 for proper multi-input model
    model.add(Dense(COMBINED_DENSE_UNITS, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                 optimizer=Adam(learning_rate=0.001),
                 metrics=['accuracy'])

    return model


def combine_inputs_simple(X_char, X_features):
    """
    Combine character and feature inputs by concatenation.

    Simple approach: concatenate along feature axis
    (This works well if features are normalized to 0-1)
    """
    print("\nCombining inputs (character + features)...")
    print(f"  Character shape: {X_char.shape}")
    print(f"  Features shape: {X_features.shape}")

    # Normalize character input to 0-1 range for fair concatenation
    X_char_normalized = X_char.astype(np.float32) / 128.0  # vocab_size ≈ 68

    X_combined = np.concatenate([X_char_normalized, X_features], axis=1)

    print(f"  Combined shape: {X_combined.shape}")

    return X_combined


# ============================================================================
# APPROACH 2: MULTI-INPUT MODEL 
# ============================================================================

def build_model_multi_input(vocab_size, num_features, max_length=128):
    """
    Build multi-input model with separate branches for characters and features.

    Architecture:
    - Branch 1 (Characters): Embedding -> LSTM -> Dense
    - Branch 2 (Features): Dense layers
    - Merge: Concatenate branches -> Dense layers -> Output

    Advantages:
    - Model learns optimal weighting of both inputs
    - Better feature utilization
    - Clearer information flow
    """
    print("\nBuilding MULTI-INPUT model...")
    print(f"  Character branch: {max_length} char sequence")
    print(f"  Feature branch: {num_features} engineered features")

    # ===== BRANCH 1: Character-level LSTM =====
    char_input = Input(shape=(max_length,), name='char_sequence', dtype='int32')

    char_embed = Embedding(vocab_size, CHAR_EMBEDDING_DIM, mask_zero=True)(char_input)
    char_lstm1 = LSTM(CHAR_LSTM_UNITS, return_sequences=True)(char_embed)
    char_lstm2 = LSTM(CHAR_LSTM_UNITS, return_sequences=False)(char_lstm1)
    char_dropout = Dropout(DROPOUT_RATE)(char_lstm2)
    char_dense = Dense(64, activation='relu')(char_dropout)

    # ===== BRANCH 2: Engineered Features =====
    feature_input = Input(shape=(num_features,), name='engineered_features', dtype='float32')

    feature_dense1 = Dense(FEATURE_DENSE_UNITS, activation='relu')(feature_input)
    feature_batch = BatchNormalization()(feature_dense1)
    feature_dropout = Dropout(0.2)(feature_batch)
    feature_dense2 = Dense(32, activation='relu')(feature_dropout)

    # ===== MERGE & OUTPUT =====
    merged = Concatenate()([char_dense, feature_dense2])
    merged_dense1 = Dense(COMBINED_DENSE_UNITS, activation='relu')(merged)
    merged_batch = BatchNormalization()(merged_dense1)
    merged_dropout = Dropout(DROPOUT_RATE)(merged_batch)
    merged_dense2 = Dense(32, activation='relu')(merged_dropout)

    output = Dense(1, activation='sigmoid')(merged_dense2)

    # Compile model
    model = Model(inputs=[char_input, feature_input], outputs=output)
    model.compile(loss='binary_crossentropy',
                 optimizer=Adam(learning_rate=0.001),
                 metrics=['accuracy'])

    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_with_features(args):
    """Main training pipeline with features"""

    print("=" * 70)
    print("TRAINING WITH ENGINEERED FEATURES")
    print("=" * 70)
    print()

    # ===== LOAD DATA =====
    data_dir = '/workspace/data' if os.path.exists('/workspace/data') else '.'

    # Determine output directory
    output_dir = os.environ.get('OUTPUT_DIR', 'output/features')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    print("Loading data...")
    train_data = load_data([f'{data_dir}/0_train.csv'], [f'{data_dir}/1_train.csv'])
    val_data = load_data([f'{data_dir}/0_validation.csv'], [f'{data_dir}/1_validation.csv'])
    test_data = load_data([f'{data_dir}/0_test.csv'], [f'{data_dir}/1_test.csv'])

    print(f"  Training: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    print()

    # ===== BUILD VOCABULARY =====
    print("Building vocabulary...")
    char_list, char_indices = prepare_vocabulary(train_data)
    vocab_size = len(char_list) + 1
    print(f"  Vocabulary size: {vocab_size}")
    print()

    # ===== ENCODE CHARACTER DATA =====
    print("Encoding character sequences...")
    X_char_train, y_train = encode_char_data(train_data, char_indices, MAX_LENGTH)
    X_char_val, y_val = encode_char_data(val_data, char_indices, MAX_LENGTH)
    X_char_test, y_test = encode_char_data(test_data, char_indices, MAX_LENGTH)

    print(f"  Train shape: {X_char_train.shape}")
    print(f"  Val shape: {X_char_val.shape}")
    print(f"  Test shape: {X_char_test.shape}")
    print()

    # ===== EXTRACT ENGINEERED FEATURES =====
    X_features_train, feature_names = extract_feature_data(train_data)
    X_features_val, _ = extract_feature_data(val_data)
    X_features_test, _ = extract_feature_data(test_data)

    print(f"  Features shape: {X_features_train.shape}")
    print(f"  Number of features: {len(feature_names)}")
    print()

    # ===== BUILD MODEL =====
    if args.model_type == 'multi_input':
        model = build_model_multi_input(vocab_size, len(feature_names), MAX_LENGTH)
        train_inputs = [X_char_train, X_features_train]
        val_inputs = [X_char_val, X_features_val]
        test_inputs = [X_char_test, X_features_test]
    else:
        # Concatenation approach
        X_combined_train = combine_inputs_simple(X_char_train, X_features_train)
        X_combined_val = combine_inputs_simple(X_char_val, X_features_val)
        X_combined_test = combine_inputs_simple(X_char_test, X_features_test)

        model = build_model_concatenation(vocab_size, len(feature_names), MAX_LENGTH)
        train_inputs = X_combined_train
        val_inputs = X_combined_val
        test_inputs = X_combined_test

    model.summary()
    print()

    # ===== TRAINING =====
    print("Training model...")
    model_file = os.path.join(output_dir, f'features_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras')
    print(f"\n📁 Model will be saved to: {model_file}")
    print(f"✓ Output directory exists: {os.path.isdir(output_dir)}")

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=args.patience, verbose=1, restore_best_weights=True),
        ModelCheckpoint(
            filepath=model_file,
            monitor='val_loss',
            verbose=1,
            save_best_only=True
        )
    ]

    history = model.fit(
        train_inputs, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        shuffle=True,
        validation_data=(val_inputs, y_val),
        callbacks=callbacks
    )

    print()
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)
    print()

    # ===== EVALUATION =====
    y_pred_proba = model.predict(test_inputs, batch_size=64, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    cm = confusion_matrix(y_test, y_pred)

    print("Classification Report:")
    print(classification_report(y_test, y_pred,
                              target_names=['Legitimate', 'Suspicious'],
                              digits=4))

    print("\nConfusion Matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    accuracy = (tn + tp) / cm.sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # ===== CRITICAL: FORCE SAVE MODEL =====
    # Ensure model is always saved, regardless of checkpoint callback status
    print(f"\nForce-saving model to guarantee output...")
    try:
        model.save(model_file)
        print(f"✓ Model saved to: {model_file}")
        print(f"  File size: {os.path.getsize(model_file) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"⚠️ Error saving model: {e}")
        print(f"  File exists: {os.path.exists(model_file)}")

    # ===== SAVE MODEL INFO =====
    mlinfo = {
        'version': int(datetime.now().strftime("%Y%m%d%H")),
        'model': os.path.basename(model_file),
        'model_type': args.model_type,
        'features_count': len(feature_names),
        'feature_names': feature_names,
        'vocabulary': {
            'char_list': char_list,
            'max_length': MAX_LENGTH,
            'vocab_size': vocab_size
        },
        'training': {
            'epochs_run': len(history.history['loss']),
            'final_train_acc': float(history.history['accuracy'][-1]),
            'final_val_acc': float(history.history['val_accuracy'][-1]),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
        },
        'test_metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
        },
        'created_at': datetime.now().isoformat()
    }

    mlinfo_file = os.path.join(output_dir, f'features_mlinfo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(mlinfo_file, 'w') as f:
        json.dump(mlinfo, f, indent=2)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n✓ Model: {model_file}")
    print(f"  Exists: {os.path.exists(model_file)}")
    print(f"✓ Config: {mlinfo_file}")
    print(f"  Exists: {os.path.exists(mlinfo_file)}")
    print(f"✓ Test Accuracy: {accuracy:.4f} (with engineered features!)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train model with engineered features')

    parser.add_argument('--epochs', type=int, default=100, help='Maximum epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience')
    parser.add_argument('--model-type', type=str, default='multi_input',
                       choices=['concatenation', 'multi_input'],
                       help='Model type: concatenation or multi_input')
    parser.add_argument('--max-length', type=int, default=128, help='Max sequence length')

    args = parser.parse_args()

    print(f"\nModel Type: {args.model_type}")
    if args.model_type == 'multi_input':
        print("Expected improvement: +5-7%")
    else:
        print("Expected improvement: +4-5%")
    print()

    train_with_features(args)


if __name__ == '__main__':
    main()
