#!/usr/bin/env python3
"""
Extract Confidence Score Distributions for All Models

This script:
1. Loads test data
2. Runs predictions for all 7 models
3. Calculates confidence score distributions (0.0-0.1, 0.1-0.2, ... 0.9-1.0)
4. Separates distributions by actual label (legitimate vs malicious)
5. Outputs structured data with percentages for each bin

Usage:
    python3 extract_confidence_distributions.py
"""

import os
import json
import glob
import pickle
import numpy as np
import csv
import codecs
from collections import defaultdict
from datetime import datetime
import sys

# Import necessary libraries
try:
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.preprocessing import sequence
    from tensorflow.keras.models import load_model, Model
    import xgboost as xgb
except ImportError as e:
    print(f"Error: {e}")
    print("Please install required dependencies")
    sys.exit(1)

# Configuration
DATA_DIR = '.'
if not os.path.exists('0_test.csv'):
    DATA_DIR = '..'
TEST_DATA_0 = os.path.join(DATA_DIR, '0_test.csv')
TEST_DATA_1 = os.path.join(DATA_DIR, '1_test.csv')
MAX_LENGTH = 128

# Determine OUTPUT_DIR based on where we're running from
if os.path.exists('models/output'):
    OUTPUT_DIR = 'models/output'
elif os.path.exists('../models/output'):
    OUTPUT_DIR = '../models/output'
else:
    OUTPUT_DIR = 'models/output'

# Import feature extraction
try:
    # Try both possible locations
    if os.path.exists('models'):
        sys.path.insert(0, 'models')
    elif os.path.exists('../models'):
        sys.path.insert(0, '../models')

    from feature_utils import extract_features_to_array
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False
    print("Warning: feature_utils not available")

# Import TransformerBlock for custom object loading
try:
    # Path already added above
    from train_transformer import TransformerBlock
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    TransformerBlock = None


def load_test_data():
    """Load test data"""
    print("Loading test data...")
    data = []

    # Load legitimate packages (label 0)
    if os.path.exists(TEST_DATA_0):
        with codecs.open(TEST_DATA_0, 'r', 'utf8') as f:
            reader = csv.reader(f)
            for line in reader:
                data.append((0, line[0]))

    # Load suspicious packages (label 1)
    if os.path.exists(TEST_DATA_1):
        with codecs.open(TEST_DATA_1, 'r', 'utf8') as f:
            reader = csv.reader(f)
            for line in reader:
                data.append((1, line[0]))

    print(f"Loaded {len(data)} test samples")
    return data


def get_vocabulary_from_mlinfo(mlinfo_path):
    """Extract vocabulary from mlinfo.json"""
    try:
        with open(mlinfo_path, 'r') as f:
            mlinfo = json.load(f)
        if 'vocabulary' in mlinfo:
            return mlinfo['vocabulary']['char_list']
        else:
            return None
    except:
        return None


def encode_data(data, char_list, max_length=128):
    """Encode package names to character indices"""
    X, y = [], []
    char_indices = dict((c, i+1) for i, c in enumerate(char_list))

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


def load_model_with_custom_objects(model_path, model_name='unknown'):
    """Load a model with appropriate custom objects"""
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, Dropout, Activation, Bidirectional

    try:
        from tensorflow.compat.v1.keras.layers import LSTM
    except:
        from tensorflow.keras.layers import LSTM

    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        custom_objects = {
            'LSTM': LSTM,
            'Bidirectional': Bidirectional,
            'Dense': Dense,
            'Embedding': Embedding,
            'Conv1D': Conv1D,
            'MaxPooling1D': MaxPooling1D,
            'Dropout': Dropout,
            'Activation': Activation
        }

        if 'transformer' in model_name.lower() and TransformerBlock is not None:
            custom_objects['TransformerBlock'] = TransformerBlock

        if model_path.endswith('.keras'):
            try:
                model = load_model(model_path)
                return model
            except:
                model = load_model(model_path, custom_objects=custom_objects)
                return model

        model = load_model(model_path, custom_objects=custom_objects)
        return model

    except Exception as e:
        try:
            model = load_model(model_path, custom_objects=custom_objects, compile=False)
            try:
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            except:
                pass
            return model
        except Exception as e2:
            raise Exception(f"Could not load model {model_path}: {str(e2)}")


def calculate_distribution(y_pred_proba, y_true):
    """
    Calculate confidence score distribution by actual label

    Returns:
        dict: Distribution for label 0 (legitimate) and label 1 (malicious)
    """
    bins = np.arange(0, 1.1, 0.1)
    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]

    # Flatten predictions
    y_pred_flat = y_pred_proba.flatten()

    # Split by actual label
    legitimate_mask = (y_true == 0)
    malicious_mask = (y_true == 1)

    legitimate_scores = y_pred_flat[legitimate_mask]
    malicious_scores = y_pred_flat[malicious_mask]

    # Calculate histograms
    legit_hist, _ = np.histogram(legitimate_scores, bins=bins)
    malicious_hist, _ = np.histogram(malicious_scores, bins=bins)

    # Convert to percentages
    legit_total = len(legitimate_scores)
    malicious_total = len(malicious_scores)

    legit_pct = (legit_hist / legit_total * 100) if legit_total > 0 else np.zeros_like(legit_hist)
    malicious_pct = (malicious_hist / malicious_total * 100) if malicious_total > 0 else np.zeros_like(malicious_hist)

    result = {
        'legitimate': {
            'total_samples': int(legit_total),
            'bins': {}
        },
        'malicious': {
            'total_samples': int(malicious_total),
            'bins': {}
        }
    }

    for i, label in enumerate(bin_labels):
        result['legitimate']['bins'][label] = {
            'count': int(legit_hist[i]),
            'percentage': float(legit_pct[i])
        }
        result['malicious']['bins'][label] = {
            'count': int(malicious_hist[i]),
            'percentage': float(malicious_pct[i])
        }

    return result


def predict_character_model(model_path, mlinfo_path, test_data_X, test_data_y, model_name):
    """Run predictions for character-based model"""
    try:
        print(f"  Loading model...")
        model = load_model_with_custom_objects(model_path, model_name)

        print(f"  Running predictions...")
        y_pred_proba = model.predict(test_data_X, batch_size=64, verbose=0)

        print(f"  Calculating distributions...")
        distribution = calculate_distribution(y_pred_proba, test_data_y)

        return {
            'model_name': model_name,
            'status': 'SUCCESS',
            'distribution': distribution
        }
    except Exception as e:
        return {
            'model_name': model_name,
            'status': 'FAILED',
            'error': str(e)
        }


def predict_features_model(model_path, mlinfo_path, test_data, test_data_y, model_name):
    """Run predictions for multi-input features model"""
    if not FEATURES_AVAILABLE:
        return {'model_name': model_name, 'status': 'SKIPPED', 'reason': 'feature_utils not available'}

    try:
        print(f"  Loading model...")
        model = load_model_with_custom_objects(model_path, model_name)

        # Load vocabulary
        with open(mlinfo_path, 'r') as f:
            mlinfo = json.load(f)

        char_list = mlinfo['vocabulary']['char_list']
        char_indices = dict((c, i+1) for i, c in enumerate(char_list))

        # Encode characters
        print(f"  Encoding characters...")
        X_char = []
        for label, package_name in test_data:
            indices = []
            for char in package_name:
                if char in char_indices:
                    indices.append(char_indices[char])
                else:
                    indices.append(0)
            padded = sequence.pad_sequences([indices], maxlen=MAX_LENGTH)[0]
            X_char.append(padded)

        X_char = np.array(X_char, dtype=np.uint8)

        # Extract features
        print(f"  Extracting features...")
        X_features = []
        for idx, (label, package_name) in enumerate(test_data):
            if idx % 5000 == 0:
                print(f"    Processed {idx}/{len(test_data)}")
            features, _ = extract_features_to_array(package_name)
            X_features.append(features)

        X_features = np.array(X_features, dtype=np.float32)

        # Normalize features
        scaler = MinMaxScaler()
        X_features = scaler.fit_transform(X_features)

        # Predict
        print(f"  Running predictions...")
        y_pred_proba = model.predict([X_char, X_features], batch_size=64, verbose=0)

        print(f"  Calculating distributions...")
        distribution = calculate_distribution(y_pred_proba, test_data_y)

        return {
            'model_name': model_name,
            'status': 'SUCCESS',
            'distribution': distribution
        }
    except Exception as e:
        return {
            'model_name': model_name,
            'status': 'FAILED',
            'error': str(e)
        }


def predict_xgboost_model(model_path, mlinfo_path, test_data, test_data_y, model_name):
    """Run predictions for XGBoost model"""
    if not FEATURES_AVAILABLE:
        return {'model_name': model_name, 'status': 'SKIPPED', 'reason': 'feature_utils not available'}

    try:
        print(f"  Loading XGBoost model...")
        with open(model_path, 'rb') as f:
            xgb_model = pickle.load(f)

        # Extract features
        print(f"  Extracting features...")
        X_test_features = []
        for idx, (label, package_name) in enumerate(test_data):
            if idx % 5000 == 0:
                print(f"    Processed {idx}/{len(test_data)}")
            features, _ = extract_features_to_array(package_name)
            X_test_features.append(features)

        X_test_features = np.array(X_test_features, dtype=np.float32)

        # Get predictions
        print(f"  Running predictions...")
        dtest = xgb.DMatrix(X_test_features)
        y_pred_proba = xgb_model.predict(dtest)

        print(f"  Calculating distributions...")
        distribution = calculate_distribution(y_pred_proba, test_data_y)

        return {
            'model_name': model_name,
            'status': 'SUCCESS',
            'distribution': distribution
        }
    except Exception as e:
        return {
            'model_name': model_name,
            'status': 'FAILED',
            'error': str(e)
        }


def find_all_models():
    """Find all trained models"""
    models = []

    # Character-based models
    for arch in ['bilstm', 'transformer', 'cnn', 'cnn_lstm', 'dummy']:
        pattern = os.path.join(OUTPUT_DIR, arch, f'{arch}_model_*.hdf5')
        matching_files = glob.glob(pattern)

        if matching_files:
            model_file = matching_files[-1]
            mlinfo_pattern = os.path.join(OUTPUT_DIR, arch, f'{arch}_mlinfo_*.json')
            mlinfo_files = glob.glob(mlinfo_pattern)

            if mlinfo_files:
                mlinfo_file = mlinfo_files[-1]
                models.append({
                    'arch': arch,
                    'model_file': model_file,
                    'mlinfo_file': mlinfo_file,
                    'model_type': 'character'
                })

    # Features model
    features_pattern = os.path.join(OUTPUT_DIR, 'features', 'features_model_*.keras')
    matching_features = glob.glob(features_pattern)
    if not matching_features:
        features_pattern = os.path.join(OUTPUT_DIR, 'features', 'features_model_*.hdf5')
        matching_features = glob.glob(features_pattern)

    if matching_features:
        model_file = matching_features[-1]
        mlinfo_pattern = os.path.join(OUTPUT_DIR, 'features', 'features_mlinfo_*.json')
        mlinfo_features = glob.glob(mlinfo_pattern)

        if mlinfo_features:
            mlinfo_file = mlinfo_features[-1]
            models.append({
                'arch': 'features',
                'model_file': model_file,
                'mlinfo_file': mlinfo_file,
                'model_type': 'features'
            })

    # XGBoost model
    xgb_pattern = os.path.join(OUTPUT_DIR, 'xgboost', 'xgboost_model_*.pkl')
    matching_xgb = glob.glob(xgb_pattern)
    if matching_xgb:
        model_file = matching_xgb[-1]
        mlinfo_pattern = os.path.join(OUTPUT_DIR, 'xgboost', 'xgboost_mlinfo_*.json')
        mlinfo_files = glob.glob(mlinfo_pattern)
        if mlinfo_files:
            mlinfo_file = mlinfo_files[-1]
            models.append({
                'arch': 'xgboost',
                'model_file': model_file,
                'mlinfo_file': mlinfo_file,
                'model_type': 'xgboost'
            })

    return models


def generate_ascii_chart(distribution, label_type):
    """Generate ASCII bar chart for distribution"""
    bins = distribution[label_type]['bins']
    total = distribution[label_type]['total_samples']

    lines = []
    lines.append(f"\n{label_type.upper()} APPS (actual label = {'0' if label_type == 'legitimate' else '1'}):")
    lines.append(f"Total samples: {total:,}")
    lines.append("-" * 70)

    max_bar_width = 50
    max_pct = max(bin_data['percentage'] for bin_data in bins.values())

    for bin_label in sorted(bins.keys()):
        bin_data = bins[bin_label]
        pct = bin_data['percentage']
        count = bin_data['count']

        # Calculate bar width
        if max_pct > 0:
            bar_width = int((pct / max_pct) * max_bar_width)
        else:
            bar_width = 0

        filled = '█' * bar_width
        empty = '░' * (max_bar_width - bar_width)

        lines.append(f"{bin_label}:  {filled}{empty}  ({pct:5.1f}% - {count:,} apps)")

    return "\n".join(lines)


def main():
    print("=" * 80)
    print("CONFIDENCE SCORE DISTRIBUTION EXTRACTION")
    print("=" * 80)
    print()

    # Load test data
    test_data = load_test_data()
    print()

    # Find all models
    models = find_all_models()

    if not models:
        print("ERROR: No trained models found!")
        return

    print(f"Found {len(models)} models:\n")
    for m in models:
        model_file = m.get('model_file', 'unknown')
        print(f"  ✓ {m['arch']}: {model_file}")
    print()

    # Get vocabulary from first character model
    first_char_model = [m for m in models if m['model_type'] == 'character']
    if first_char_model:
        vocab = get_vocabulary_from_mlinfo(first_char_model[0]['mlinfo_file'])
    else:
        vocab = None

    if not vocab and any(m['model_type'] == 'character' for m in models):
        print("ERROR: Could not extract vocabulary from mlinfo!")
        return

    # Encode test data for character models
    if first_char_model and vocab:
        print(f"Using vocabulary with {len(vocab)} characters")
        X_test, y_test = encode_data(test_data, vocab, MAX_LENGTH)
        print(f"Test set shape: {X_test.shape}\n")
    else:
        X_test, y_test = None, None

    # Run predictions for all models
    print("Extracting confidence distributions...\n")
    results = {}

    for model_info in models:
        arch = model_info['arch']
        print(f"Processing {arch}...")

        if model_info['model_type'] == 'character':
            result = predict_character_model(
                model_info['model_file'],
                model_info['mlinfo_file'],
                X_test,
                y_test,
                arch
            )
        elif model_info['model_type'] == 'features':
            result = predict_features_model(
                model_info['model_file'],
                model_info['mlinfo_file'],
                test_data,
                y_test,
                arch
            )
        elif model_info['model_type'] == 'xgboost':
            result = predict_xgboost_model(
                model_info['model_file'],
                model_info['mlinfo_file'],
                test_data,
                y_test,
                arch
            )
        else:
            result = {'status': 'SKIPPED', 'reason': f'Unknown model type: {model_info["model_type"]}'}

        if result['status'] == 'SUCCESS':
            print(f"  ✓ Success\n")
            results[arch] = result['distribution']
        elif result['status'] == 'SKIPPED':
            print(f"  ⊘ Skipped: {result.get('reason', 'Unknown')}\n")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}\n")

    # Save results to JSON
    output_file = f'confidence_distributions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nSaved detailed distributions to: {output_file}")

    # Generate and display ASCII charts
    print("\n" + "=" * 80)
    print("DISTRIBUTION VISUALIZATIONS")
    print("=" * 80)

    for model_name in sorted(results.keys()):
        print(f"\n{'=' * 80}")
        print(f"MODEL: {model_name.upper()}")
        print('=' * 80)

        distribution = results[model_name]

        # Show legitimate apps distribution
        print(generate_ascii_chart(distribution, 'legitimate'))

        # Show malicious apps distribution
        print(generate_ascii_chart(distribution, 'malicious'))
        print()

    print("\n" + "=" * 80)
    print("COMPLETED")
    print("=" * 80)


if __name__ == '__main__':
    main()
