#!/usr/bin/env python3
"""
Evaluate a single model in isolation (designed to run in Docker)

This script allows evaluating individual models in Docker containers
to isolate TensorFlow versions and eliminate compatibility issues.

Usage:
    python3 evaluate_model_in_docker.py --model <model_name> --output <output_dir>

Example:
    python3 evaluate_model_in_docker.py --model bilstm --output /workspace/results
"""

import os
import json
import glob
import pickle
import numpy as np
from datetime import datetime
import csv
import codecs
import argparse
from collections import Counter

# Configuration
DATA_DIR = '/workspace/data'
MODELS_DIR = '/workspace/models'
OUTPUT_DIR = '/workspace/output'
MAX_LENGTH = 128
TEST_DATA_0 = os.path.join(DATA_DIR, '0_test.csv')
TEST_DATA_1 = os.path.join(DATA_DIR, '1_test.csv')

# Import evaluation functions
try:
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
    from sklearn.calibration import calibration_curve
    from tensorflow.keras.preprocessing import sequence
    from tensorflow.keras.models import load_model
    import xgboost as xgb
except ImportError as e:
    print(f"Error: {e}")
    exit(1)


def load_test_data():
    """Load test data"""
    print(f"Loading test data from {DATA_DIR}...")
    data = []

    # Load legitimate packages
    if os.path.exists(TEST_DATA_0):
        with codecs.open(TEST_DATA_0, 'r', 'utf8') as f:
            reader = csv.reader(f)
            for line in reader:
                data.append((0, line[0]))
        print(f"  Loaded {sum(1 for d in data if d[0] == 0)} legitimate packages")

    # Load suspicious packages
    if os.path.exists(TEST_DATA_1):
        with codecs.open(TEST_DATA_1, 'r', 'utf8') as f:
            reader = csv.reader(f)
            for line in reader:
                data.append((1, line[0]))
        print(f"  Loaded {sum(1 for d in data if d[0] == 1)} suspicious packages")

    print(f"Total test samples: {len(data)}")
    return data


def encode_data(data, char_list, max_length):
    """Encode character sequences"""
    print(f"Encoding {len(data)} sequences...")
    char_indices = dict((c, i+1) for i, c in enumerate(char_list))

    X = []
    y = []

    for idx, (label, package_name) in enumerate(data):
        if idx % 5000 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(data)}")

        indices = []
        for char in package_name:
            if char in char_indices:
                indices.append(char_indices[char])
            else:
                indices.append(0)

        padded = sequence.pad_sequences([indices], maxlen=max_length)[0]
        X.append(padded)
        y.append(label)

    return np.array(X, dtype=np.uint8), np.array(y)


def evaluate_model(model_name, model_path, mlinfo_path, test_data, test_data_X, test_data_y):
    """Evaluate a single model"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"Model: {model_path}")
    print(f"MLInfo: {mlinfo_path}")
    print(f"{'='*80}\n")

    try:
        # Load mlinfo
        with open(mlinfo_path, 'r') as f:
            mlinfo = json.load(f)

        print(f"Model version: {mlinfo.get('version', 'unknown')}")
        print(f"Model file: {mlinfo.get('model', 'unknown')}")

        # Load model with custom objects support
        print(f"Loading model: {model_path}")

        # Detect model type
        is_xgboost = model_path.endswith('.pkl') or model_name == 'xgboost'

        if is_xgboost:
            # Load XGBoost model
            print(f"  Detected XGBoost model (pickle format)")
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"✓ XGBoost model loaded successfully")

        else:
            # Load Keras/TensorFlow model with custom objects support
            # PYTHONPATH is set to /workspace/models in Dockerfile
            custom_objects = {}
            try:
                from custom_layers import TransformerBlock, CNNLSTM
                custom_objects['TransformerBlock'] = TransformerBlock
                custom_objects['CNNLSTM'] = CNNLSTM
                print("  ✓ Loaded TransformerBlock and CNNLSTM custom objects")
            except Exception as e:
                print(f"  ℹ Custom objects import failed: {e}")

            # Load model with custom objects
            try:
                print(f"  Attempting load with custom_objects: {list(custom_objects.keys())}")
                model = load_model(model_path, custom_objects=custom_objects if custom_objects else None)
            except Exception as e:
                # Fallback: try with compile=False
                print(f"  Initial load failed: {str(e)[:100]}")
                print(f"  Retrying with compile=False and custom_objects...")
                model = load_model(model_path, compile=False, custom_objects=custom_objects if custom_objects else None)
                try:
                    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                except Exception as ce:
                    print(f"  Could not recompile model: {str(ce)[:100]}")

            print(f"✓ Model loaded successfully")
            print(f"  Input shape: {model.input_shape}")
            print(f"  Output shape: {model.output_shape}")

        # Prepare data for prediction
        if is_xgboost:
            # XGBoost models expect engineered features
            print(f"  Preparing engineered features for XGBoost...")
            try:
                from feature_utils import extract_all_features
                from sklearn.preprocessing import MinMaxScaler

                # Extract features from package names
                test_features = []
                feature_names = None
                for _, package_name in test_data:
                    features_dict = extract_all_features(package_name)
                    # CRITICAL: Sort feature names for consistent ordering (matches training)
                    if feature_names is None:
                        feature_names = sorted(features_dict.keys())
                    # Convert dict to sorted list of values
                    feature_list = [features_dict[fname] for fname in feature_names]
                    test_features.append(feature_list)

                # Convert to numpy array (NO NORMALIZATION - model was trained on raw features)
                test_features = np.array(test_features, dtype=np.float32)

                print(f"  Extracted features shape: {test_features.shape}")
                prediction_data = test_features
            except Exception as e:
                print(f"  ⚠️  Could not extract features: {e}")
                raise
        else:
            # Keras/TensorFlow models
            # Check if this is a multi-input model (like feature-enhanced LSTM)
            is_multi_input = isinstance(model.input_shape, list) and len(model.input_shape) > 1

            if is_multi_input:
                print(f"  Detected multi-input model - extracting engineered features...")
                # Try to load feature extraction utilities
                try:
                    from feature_utils import extract_all_features
                    from sklearn.preprocessing import MinMaxScaler

                    # Extract features from package names
                    test_features = []
                    feature_names = None
                    for _, package_name in test_data:
                        features_dict = extract_all_features(package_name)
                        # CRITICAL: Sort feature names for consistent ordering (matches training)
                        if feature_names is None:
                            feature_names = sorted(features_dict.keys())
                        # Convert dict to sorted list of values
                        feature_list = [features_dict[fname] for fname in feature_names]
                        test_features.append(feature_list)

                    # Convert to numpy array (NO NORMALIZATION - model was trained on raw features)
                    test_features = np.array(test_features, dtype=np.float32)

                    print(f"  Extracted features shape: {test_features.shape}")
                    prediction_data = [test_data_X, test_features]
                except Exception as e:
                    print(f"  ⚠️  Could not extract features: {e}")
                    print(f"  Attempting prediction with sequence data only...")
                    prediction_data = test_data_X
            else:
                prediction_data = test_data_X

        # Predict
        print(f"Running predictions on {len(test_data_X) if not is_xgboost else prediction_data.shape[0]} samples...")
        if is_xgboost:
            # XGBoost requires DMatrix format
            import xgboost as xgb
            dmatrix = xgb.DMatrix(prediction_data)
            y_pred_proba = model.predict(dmatrix)
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            # Keras model
            y_pred_proba = model.predict(prediction_data, batch_size=64, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Calculate metrics
        accuracy = accuracy_score(test_data_y, y_pred)
        precision = precision_score(test_data_y, y_pred, zero_division=0)
        recall = recall_score(test_data_y, y_pred, zero_division=0)
        f1 = f1_score(test_data_y, y_pred, zero_division=0)

        cm = confusion_matrix(test_data_y, y_pred)
        tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0, 0, 0, 0)

        # ROC-AUC
        try:
            roc_auc = roc_auc_score(test_data_y, y_pred_proba)
        except:
            roc_auc = 0.0

        # Model size
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

        result = {
            'model_name': model_name,
            'model_path': model_path,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'model_size_mb': float(model_size_mb),
            'status': 'SUCCESS',
            'timestamp': datetime.now().isoformat()
        }

        print(f"\n✓ EVALUATION RESULTS:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  Model Size: {model_size_mb:.2f} MB")

        return result

    except Exception as e:
        print(f"\n✗ EVALUATION FAILED:")
        print(f"  Error: {str(e)}")

        return {
            'model_name': model_name,
            'model_path': model_path,
            'status': 'FAILED',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def find_model(model_name):
    """Find model files in the models directory

    Strategy: Find the most recent mlinfo file first (source of truth),
    then load the model file referenced in it, or fall back to the most recent model
    """
    print(f"Searching for model: {model_name}")

    model_dir = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_dir):
        return None, None

    # First, try to find mlinfo files and use them to determine the latest model
    mlinfo_pattern = os.path.join(model_dir, f'{model_name}_mlinfo_*.json')
    mlinfo_matches = glob.glob(mlinfo_pattern)

    # Filter out 'latest' symlinks
    valid_mlinfo = [m for m in mlinfo_matches if '_latest' not in os.path.basename(m)]

    if valid_mlinfo:
        # Sort by timestamp in filename (YYYYMMDD_HHMMSS format)
        # This is more reliable than filesystem modification time
        mlinfo_matches_sorted = sorted(valid_mlinfo)
        mlinfo_file = mlinfo_matches_sorted[-1]  # Get the last one (newest by timestamp)
        print(f"  Found mlinfo: {mlinfo_file}")

        # Try to load the referenced model from mlinfo
        try:
            with open(mlinfo_file, 'r') as f:
                mlinfo = json.load(f)
                if 'model' in mlinfo:
                    referenced_model = os.path.join(model_dir, mlinfo['model'])
                    if os.path.exists(referenced_model):
                        print(f"  Found model: {referenced_model}")
                        return referenced_model, mlinfo_file
        except Exception as e:
            print(f"  Warning: Could not load mlinfo: {e}")

    # Fallback: search for model files directly
    patterns = [
        os.path.join(model_dir, f'{model_name}_model_*.hdf5'),
        os.path.join(model_dir, f'{model_name}_model_*.keras'),
        os.path.join(model_dir, f'{model_name}_model_*.pkl'),
    ]

    model_file = None

    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            # Filter out 'latest' and sort by timestamp in filename
            valid_matches = [m for m in matches if '_latest' not in os.path.basename(m)]
            if valid_matches:
                model_file = sorted(valid_matches)[-1]  # Most recent by timestamp
            else:
                model_file = sorted(matches)[-1]
            print(f"  Found model: {model_file}")
            break

    # If we still need mlinfo, find the most recent one
    if not mlinfo_file and model_file:
        mlinfo_pattern = os.path.join(model_dir, f'{model_name}_mlinfo_*.json')
        mlinfo_matches = glob.glob(mlinfo_pattern)
        if mlinfo_matches:
            valid_mlinfo = [m for m in mlinfo_matches if '_latest' not in os.path.basename(m)]
            if valid_mlinfo:
                mlinfo_file = sorted(valid_mlinfo)[-1]
                print(f"  Found mlinfo: {mlinfo_file}")

    return model_file, mlinfo_file


def main():
    parser = argparse.ArgumentParser(description='Evaluate a model in Docker')
    parser.add_argument('--model', required=True, help='Model name (e.g., bilstm, cnn, dummy)')
    parser.add_argument('--output', default=OUTPUT_DIR, help='Output directory for results')
    args = parser.parse_args()

    print(f"{'='*80}")
    print(f"MODEL EVALUATION IN DOCKER")
    print(f"TensorFlow with native model compatibility")
    print(f"{'='*80}\n")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load test data
    test_data = load_test_data()
    test_data_y = np.array([d[0] for d in test_data])

    # Find model
    model_file, mlinfo_file = find_model(args.model)

    if not model_file or not mlinfo_file:
        print(f"✗ Model '{args.model}' not found in {MODELS_DIR}")
        return False

    # Load mlinfo to get vocabulary
    with open(mlinfo_file, 'r') as f:
        mlinfo = json.load(f)

    # Detect if this is an XGBoost model
    is_xgboost = model_file.endswith('.pkl') or args.model == 'xgboost'

    # Get vocabulary (not needed for XGBoost)
    if not is_xgboost:
        if 'vocabulary' in mlinfo:
            char_list = mlinfo['vocabulary']['char_list']
        else:
            print("✗ Vocabulary not found in mlinfo")
            return False
        # Encode test data
        test_data_X, test_data_y = encode_data(test_data, char_list, MAX_LENGTH)
    else:
        # For XGBoost, we don't encode sequences
        print("  XGBoost model detected - skipping sequence encoding")
        test_data_X = None  # Will be created during evaluation
        test_data_y = np.array([d[0] for d in test_data])

    # Evaluate
    result = evaluate_model(args.model, model_file, mlinfo_file, test_data, test_data_X, test_data_y)

    # Save result
    result_file = os.path.join(args.output, f'{args.model}_evaluation_result.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Results saved to: {result_file}")

    return result['status'] == 'SUCCESS'


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
