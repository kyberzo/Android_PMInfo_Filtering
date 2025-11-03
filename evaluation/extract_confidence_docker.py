#!/usr/bin/env python3
"""
Extract Confidence Score Distributions - Docker Version

This script is designed to run inside the evaluation Docker container
where all models can be loaded properly.

Usage (from host):
    docker run -v $(pwd)/..:/workspace -w /workspace/evaluation \
        tf-evaluation:latest python3 extract_confidence_docker.py
"""

import os
import json
import glob
import pickle
import numpy as np
import csv
import codecs
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
    sys.exit(1)

# Configuration for Docker environment
DATA_DIR = '/workspace'
TEST_DATA_0 = os.path.join(DATA_DIR, '0_test.csv')
TEST_DATA_1 = os.path.join(DATA_DIR, '1_test.csv')
MAX_LENGTH = 128
OUTPUT_DIR = '/workspace/models/output'

# Import feature extraction
sys.path.insert(0, '/workspace/models')
try:
    from feature_utils import extract_features_to_array
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False

# Import TransformerBlock
try:
    from train_transformer import TransformerBlock
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    TransformerBlock = None


def load_test_data():
    """Load test data"""
    print("Loading test data...")
    data = []

    if os.path.exists(TEST_DATA_0):
        with codecs.open(TEST_DATA_0, 'r', 'utf8') as f:
            reader = csv.reader(f)
            for line in reader:
                data.append((0, line[0]))

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


def load_model_safe(model_path, model_name='unknown'):
    """Load model with error handling"""
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, Dropout, Activation, Bidirectional, LSTM

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

        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    except Exception as e:
        raise Exception(f"Could not load model: {str(e)}")


def calculate_distribution(y_pred_proba, y_true):
    """Calculate confidence score distribution by actual label"""
    bins = np.arange(0, 1.1, 0.1)
    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]

    y_pred_flat = y_pred_proba.flatten()
    legitimate_mask = (y_true == 0)
    malicious_mask = (y_true == 1)

    legitimate_scores = y_pred_flat[legitimate_mask]
    malicious_scores = y_pred_flat[malicious_mask]

    legit_hist, _ = np.histogram(legitimate_scores, bins=bins)
    malicious_hist, _ = np.histogram(malicious_scores, bins=bins)

    legit_total = len(legitimate_scores)
    malicious_total = len(malicious_scores)

    legit_pct = (legit_hist / legit_total * 100) if legit_total > 0 else np.zeros_like(legit_hist)
    malicious_pct = (malicious_hist / malicious_total * 100) if malicious_total > 0 else np.zeros_like(malicious_hist)

    result = {
        'legitimate': {'total_samples': int(legit_total), 'bins': {}},
        'malicious': {'total_samples': int(malicious_total), 'bins': {}}
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


def predict_character_model(model_path, test_data_X, test_data_y, model_name):
    """Run predictions for character-based model"""
    try:
        print(f"  Loading {model_name}...")
        model = load_model_safe(model_path, model_name)

        print(f"  Predicting...")
        y_pred_proba = model.predict(test_data_X, batch_size=64, verbose=0)

        print(f"  Calculating distribution...")
        distribution = calculate_distribution(y_pred_proba, test_data_y)

        return {'model_name': model_name, 'status': 'SUCCESS', 'distribution': distribution}
    except Exception as e:
        return {'model_name': model_name, 'status': 'FAILED', 'error': str(e)}


def predict_features_model(model_path, mlinfo_path, test_data, test_data_y, model_name):
    """Run predictions for features model"""
    if not FEATURES_AVAILABLE:
        return {'model_name': model_name, 'status': 'SKIPPED', 'reason': 'features not available'}

    try:
        print(f"  Loading {model_name}...")
        model = load_model_safe(model_path, model_name)

        with open(mlinfo_path, 'r') as f:
            mlinfo = json.load(f)

        char_list = mlinfo['vocabulary']['char_list']
        char_indices = dict((c, i+1) for i, c in enumerate(char_list))

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

        print(f"  Extracting features...")
        X_features = []
        for idx, (label, package_name) in enumerate(test_data):
            if idx % 5000 == 0:
                print(f"    {idx}/{len(test_data)}")
            features, _ = extract_features_to_array(package_name)
            X_features.append(features)

        X_features = np.array(X_features, dtype=np.float32)
        scaler = MinMaxScaler()
        X_features = scaler.fit_transform(X_features)

        print(f"  Predicting...")
        y_pred_proba = model.predict([X_char, X_features], batch_size=64, verbose=0)

        print(f"  Calculating distribution...")
        distribution = calculate_distribution(y_pred_proba, test_data_y)

        return {'model_name': model_name, 'status': 'SUCCESS', 'distribution': distribution}
    except Exception as e:
        return {'model_name': model_name, 'status': 'FAILED', 'error': str(e)}


def predict_xgboost_model(model_path, test_data, test_data_y, model_name):
    """Run predictions for XGBoost model"""
    if not FEATURES_AVAILABLE:
        return {'model_name': model_name, 'status': 'SKIPPED', 'reason': 'features not available'}

    try:
        print(f"  Loading {model_name}...")
        with open(model_path, 'rb') as f:
            xgb_model = pickle.load(f)

        print(f"  Extracting features...")
        X_test_features = []
        for idx, (label, package_name) in enumerate(test_data):
            if idx % 5000 == 0:
                print(f"    {idx}/{len(test_data)}")
            features, _ = extract_features_to_array(package_name)
            X_test_features.append(features)

        X_test_features = np.array(X_test_features, dtype=np.float32)

        print(f"  Predicting...")
        dtest = xgb.DMatrix(X_test_features)
        y_pred_proba = xgb_model.predict(dtest)

        print(f"  Calculating distribution...")
        distribution = calculate_distribution(y_pred_proba, test_data_y)

        return {'model_name': model_name, 'status': 'SUCCESS', 'distribution': distribution}
    except Exception as e:
        return {'model_name': model_name, 'status': 'FAILED', 'error': str(e)}


def find_models():
    """Find all trained models"""
    models = []

    for arch in ['bilstm', 'transformer', 'cnn', 'cnn_lstm', 'dummy']:
        pattern = os.path.join(OUTPUT_DIR, arch, f'{arch}_model_*.hdf5')
        matching = glob.glob(pattern)
        if matching:
            mlinfo_pattern = os.path.join(OUTPUT_DIR, arch, f'{arch}_mlinfo_*.json')
            mlinfo = glob.glob(mlinfo_pattern)
            if mlinfo:
                models.append({
                    'arch': arch,
                    'model_file': matching[-1],
                    'mlinfo_file': mlinfo[-1],
                    'model_type': 'character'
                })

    # Features model
    for ext in ['.keras', '.hdf5']:
        pattern = os.path.join(OUTPUT_DIR, 'features', f'features_model_*{ext}')
        matching = glob.glob(pattern)
        if matching:
            mlinfo_pattern = os.path.join(OUTPUT_DIR, 'features', 'features_mlinfo_*.json')
            mlinfo = glob.glob(mlinfo_pattern)
            if mlinfo:
                models.append({
                    'arch': 'features',
                    'model_file': matching[-1],
                    'mlinfo_file': mlinfo[-1],
                    'model_type': 'features'
                })
                break

    # XGBoost
    pattern = os.path.join(OUTPUT_DIR, 'xgboost', 'xgboost_model_*.pkl')
    matching = glob.glob(pattern)
    if matching:
        mlinfo_pattern = os.path.join(OUTPUT_DIR, 'xgboost', 'xgboost_mlinfo_*.json')
        mlinfo = glob.glob(mlinfo_pattern)
        if mlinfo:
            models.append({
                'arch': 'xgboost',
                'model_file': matching[-1],
                'mlinfo_file': mlinfo[-1],
                'model_type': 'xgboost'
            })

    return models


def main():
    print("=" * 80)
    print("CONFIDENCE DISTRIBUTION EXTRACTION (Docker Version)")
    print("=" * 80)
    print()

    test_data = load_test_data()
    print()

    models = find_models()
    if not models:
        print("ERROR: No models found!")
        return

    print(f"Found {len(models)} models\n")

    first_char = [m for m in models if m['model_type'] == 'character']
    if first_char:
        vocab = get_vocabulary_from_mlinfo(first_char[0]['mlinfo_file'])
        if vocab:
            print(f"Vocabulary: {len(vocab)} characters")
            X_test, y_test = encode_data(test_data, vocab, MAX_LENGTH)
            print(f"Test shape: {X_test.shape}\n")
        else:
            print("ERROR: No vocabulary found")
            return
    else:
        X_test, y_test = None, None

    results = {}

    for model_info in models:
        arch = model_info['arch']
        print(f"\nProcessing {arch}...")

        if model_info['model_type'] == 'character':
            result = predict_character_model(
                model_info['model_file'], X_test, y_test, arch
            )
        elif model_info['model_type'] == 'features':
            result = predict_features_model(
                model_info['model_file'], model_info['mlinfo_file'],
                test_data, y_test, arch
            )
        elif model_info['model_type'] == 'xgboost':
            result = predict_xgboost_model(
                model_info['model_file'], test_data, y_test, arch
            )

        if result['status'] == 'SUCCESS':
            print(f"  ✓ Success")
            results[arch] = result['distribution']
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown')}")

    output_file = '/workspace/evaluation/confidence_distributions_all_models.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nSaved to: {output_file}")
    print(f"Extracted distributions for {len(results)} models")


if __name__ == '__main__':
    main()
