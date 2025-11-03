"""
Shared data loading and preprocessing utilities for all model experiments
"""

import csv
import codecs
import numpy as np
from tensorflow.keras.preprocessing import sequence


def load_data(label_0_files, label_1_files):
    """Load and combine data from CSV files"""
    data = []

    for filename in label_0_files:
        try:
            with codecs.open(filename, 'r', 'utf8') as f:
                reader = csv.reader(f)
                for line in reader:
                    data.append((0, line[0]))  # (label, package_name)
        except FileNotFoundError:
            print(f"Warning: {filename} not found, skipping...")

    for filename in label_1_files:
        try:
            with codecs.open(filename, 'r', 'utf8') as f:
                reader = csv.reader(f)
                for line in reader:
                    data.append((1, line[0]))
        except FileNotFoundError:
            print(f"Warning: {filename} not found, skipping...")

    return data


def prepare_vocabulary(training_data):
    """Create sorted vocabulary from training data"""
    all_chars = ''.join([pkg for _, pkg in training_data])
    char_list = sorted(list(set(all_chars)))
    char_indices = dict((c, i+1) for i, c in enumerate(char_list))
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


def load_and_prepare_data(max_length=128, data_dir='/workspace/data'):
    """
    Complete data loading and preparation pipeline

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test, char_list, vocab_size)
    """
    print("Loading datasets...")

    train_data = load_data(
        [f'{data_dir}/0_train.csv'],
        [f'{data_dir}/1_train.csv']
    )
    val_data = load_data(
        [f'{data_dir}/0_validation.csv'],
        [f'{data_dir}/1_validation.csv']
    )
    test_data = load_data(
        [f'{data_dir}/0_test.csv'],
        [f'{data_dir}/1_test.csv']
    )

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    # Build vocabulary from training data only
    print("\nBuilding vocabulary...")
    char_list, char_indices = prepare_vocabulary(train_data)
    vocab_size = len(char_list) + 1

    print(f"Vocabulary size: {vocab_size} ({len(char_list)} unique chars + padding)")

    # Encode datasets
    print("\nEncoding datasets...")
    X_train, y_train = encode_data(train_data, char_indices, max_length)
    X_val, y_val = encode_data(val_data, char_indices, max_length)
    X_test, y_test = encode_data(test_data, char_indices, max_length)

    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, char_list, vocab_size


def save_mlinfo(model_file, char_list, vocab_size, max_length, training_history, output_file='mlinfo.json'):
    """Save model info with embedded vocabulary"""
    import json
    from datetime import datetime

    version = int(datetime.now().strftime("%Y%m%d%H"))

    mlinfo = {
        'version': version,
        'model': model_file,
        'vocabulary': {
            'char_list': char_list,
            'max_length': max_length,
            'vocab_size': vocab_size
        },
        'training': {
            'epochs_run': len(training_history['loss']),
            'final_train_acc': float(training_history['accuracy'][-1]),
            'final_val_acc': float(training_history['val_accuracy'][-1]),
            'final_train_loss': float(training_history['loss'][-1]),
            'final_val_loss': float(training_history['val_loss'][-1]),
            'best_val_loss': float(min(training_history['val_loss']))
        },
        'created_at': datetime.now().isoformat()
    }

    with open(output_file, 'w') as f:
        json.dump(mlinfo, f, indent=2)

    return mlinfo
