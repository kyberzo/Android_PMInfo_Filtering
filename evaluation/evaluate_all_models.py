#!/usr/bin/env python
"""
Enhanced Model Evaluation and Comparison Script

Compares ALL trained models including:
1. Dummy LSTM (character-level - benchmark)
2. BiLSTM
3. Transformer
4. CNN
5. CNN+LSTM Hybrid
6. Feature Engineering Model (Multi-input with 21 features)
7. XGBoost (Gradient boosting alternative)
8. XGBoost + LSTM Stacked (BEST!) ← LSTM features + XGBoost meta-learner

Expected improvements:
- BiLSTM: +2-3%
- Transformer: +4-6%
- CNN: 10x faster
- CNN+LSTM: +1-3%
- Features Model: +5-7%
- XGBoost: +3-5%
- Stacked (LSTM+XGBoost): +6-8% ← BEST FOR THIS TASK
"""

import os
import json
import glob
import pickle
import numpy as np
from datetime import datetime
import csv
import codecs
from collections import Counter

# Import evaluation functions
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
    from sklearn.calibration import calibration_curve
    from tensorflow.keras.preprocessing import sequence
    from tensorflow.keras.models import load_model, Model
    import xgboost as xgb
except ImportError as e:
    print(f"Warning: {e}")
    print("Some dependencies may not be installed")

# Configuration
# Look for test data in parent directory (where training data is stored)
DATA_DIR = '.'
if not os.path.exists('0_test.csv'):
    DATA_DIR = '..'  # Parent directory if running from root
TEST_DATA_0 = os.path.join(DATA_DIR, '0_test.csv')
TEST_DATA_1 = os.path.join(DATA_DIR, '1_test.csv')
MAX_LENGTH = 128
OUTPUT_DIR = 'models/output'

# Try to import feature extraction from models directory
try:
    import sys
    sys.path.insert(0, 'models')
    from feature_utils import extract_all_features, extract_features_to_array
    FEATURES_AVAILABLE = True
except ImportError:
    print("Warning: feature_utils not available - features model evaluation will be skipped")
    FEATURES_AVAILABLE = False

# Import TransformerBlock for custom object loading
try:
    sys.path.insert(0, 'models')
    from train_transformer import TransformerBlock
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    TransformerBlock = None


# ============================================================================
# DATA LOADING
# ============================================================================

def load_test_data():
    """Load test data"""
    print("Loading test data...")
    data = []

    # Load legitimate packages
    if os.path.exists(TEST_DATA_0):
        with codecs.open(TEST_DATA_0, 'r', 'utf8') as f:
            reader = csv.reader(f)
            for line in reader:
                data.append((0, line[0]))

    # Load suspicious packages
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


# ============================================================================
# ADDITIONAL EVALUATION METRICS
# ============================================================================

def calculate_roc_auc(y_true, y_pred_proba):
    """Calculate ROC-AUC score"""
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        return float(roc_auc)
    except:
        return None


def calculate_calibration_metrics(y_true, y_pred_proba):
    """Calculate calibration and confidence metrics"""
    try:
        # Confidence = max probability assigned by model
        confidences = np.max([1 - y_pred_proba, y_pred_proba], axis=0)
        mean_confidence = np.mean(confidences)

        # Calibration: Expected Calibration Error (ECE)
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba.flatten(), n_bins=10)
        calibration_error = np.mean(np.abs(prob_true - prob_pred))

        # Calibration score (higher is better, 1.0 is perfect)
        calibration_score = 1.0 - calibration_error

        return {
            'mean_confidence': float(mean_confidence),
            'calibration_error': float(calibration_error),
            'calibration_score': float(calibration_score)
        }
    except:
        return {
            'mean_confidence': None,
            'calibration_error': None,
            'calibration_score': None
        }


def calculate_cost_sensitive_metrics(y_true, y_pred, cost_fn=2.0, cost_fp=1.0):
    """
    Calculate cost-sensitive evaluation metrics
    cost_fn: Cost of false negative (missing a threat) - higher weight for security
    cost_fp: Cost of false positive (false alarm)
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0, 0, 0, 0)

    # Total cost
    total_cost = (fn * cost_fn) + (fp * cost_fp)

    # Normalized cost (cost per sample)
    total_samples = len(y_true)
    cost_per_sample = total_cost / total_samples if total_samples > 0 else 0

    # Weighted metrics
    weighted_accuracy = 1.0 - (cost_per_sample / max(cost_fn, cost_fp))

    return {
        'total_cost': float(total_cost),
        'cost_per_sample': float(cost_per_sample),
        'weighted_accuracy': float(max(0, weighted_accuracy))
    }


def calculate_overfitting_metrics(model, X_test, y_test, batch_size=64):
    """
    Calculate overfitting indicators by analyzing model outputs
    Note: We can't access training history, but we can look at prediction patterns
    """
    try:
        y_pred_proba = model.predict(X_test, batch_size=batch_size, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Check if model is overly confident (potential sign of overfitting)
        confidences = np.max([1 - y_pred_proba, y_pred_proba], axis=0)
        overconfidence_ratio = np.mean(confidences > 0.95)  # Fraction of predictions > 95% confident

        # Prediction margin (distance from decision boundary 0.5)
        margins = np.abs(y_pred_proba.flatten() - 0.5)
        mean_margin = np.mean(margins)

        return {
            'overconfidence_ratio': float(overconfidence_ratio),
            'mean_prediction_margin': float(mean_margin)
        }
    except:
        return {
            'overconfidence_ratio': None,
            'mean_prediction_margin': None
        }


# ============================================================================
# ROBUST MODEL LOADING
# ============================================================================

def load_model_with_custom_objects(model_path, model_name='unknown'):
    """
    Load a model with appropriate custom objects and error handling.
    Handles TensorFlow 1.x vs 2.x compatibility issues.
    Supports both HDF5 (.hdf5, .h5) and native Keras (.keras) formats.
    """
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, Dropout, Activation, Bidirectional
    # Use TF 1.x compatible LSTM for backward compatibility with HDF5 models
    try:
        from tensorflow.compat.v1.keras.layers import LSTM
    except:
        from tensorflow.keras.layers import LSTM

    try:
        # Check if file exists first
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Build custom objects dictionary with all standard layers
        # Use TF1.x LSTM for backward compatibility with old HDF5 models
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

        # Add TransformerBlock if available and model is transformer
        if 'transformer' in model_name.lower() and TransformerBlock is not None:
            custom_objects['TransformerBlock'] = TransformerBlock

        # For .keras files (native Keras format), try both native and HDF5 formats
        if model_path.endswith('.keras'):
            print(f"  Loading native Keras format (.keras)...")
            try:
                model = load_model(model_path)
                print(f"  ✓ Loaded successfully as native Keras format")
                return model
            except Exception as e:
                # If native loading fails, it might be HDF5 saved with .keras extension
                print(f"  Native load failed ({str(e)[:50]}...), trying HDF5 loading...")
                try:
                    model = load_model(model_path, custom_objects=custom_objects)
                    print(f"  ✓ Loaded successfully as HDF5 format")
                    return model
                except Exception as e2:
                    print(f"  HDF5 load also failed ({str(e2)[:50]}...), trying compile=False...")
                    model = load_model(model_path, custom_objects=custom_objects, compile=False)
                    try:
                        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                    except:
                        pass
                    return model

        # Try loading HDF5 with custom objects
        print(f"  Loading with custom objects (TF1.x compatibility)...")
        model = load_model(model_path, custom_objects=custom_objects)

        return model

    except Exception as e:
        # If standard loading fails, try with safe mode for TF2 compatibility
        try:
            print(f"  Standard loading failed ({str(e)[:50]}...), trying compile=False mode...")

            # For HDF5 files trained with TF1.x, try using compile=False
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

            # Load without compiling to avoid custom layer/optimizer issues
            model = load_model(model_path, custom_objects=custom_objects, compile=False)

            # Recompile with standard optimizer if needed
            try:
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            except:
                pass  # Some models may not be compilable

            print(f"  ✓ Loaded successfully with compile=False")
            return model

        except Exception as e2:
            print(f"  ERROR: Failed to load model: {str(e2)[:100]}")
            raise Exception(f"Could not load model {model_path}: {str(e2)}")


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_character_model(model_path, mlinfo_path, test_data_X, test_data_y, model_name):
    """Evaluate a character-based model (all except features model)"""
    try:
        model = load_model_with_custom_objects(model_path, model_name)
        y_pred_proba = model.predict(test_data_X, batch_size=64, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Calculate standard metrics
        accuracy = accuracy_score(test_data_y, y_pred)
        precision = precision_score(test_data_y, y_pred, zero_division=0)
        recall = recall_score(test_data_y, y_pred, zero_division=0)
        f1 = f1_score(test_data_y, y_pred, zero_division=0)

        cm = confusion_matrix(test_data_y, y_pred)
        tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0, 0, 0, 0)

        # Calculate additional metrics
        roc_auc = calculate_roc_auc(test_data_y, y_pred_proba)
        calibration = calculate_calibration_metrics(test_data_y, y_pred_proba)
        cost_sensitive = calculate_cost_sensitive_metrics(test_data_y, y_pred)
        overfitting = calculate_overfitting_metrics(model, test_data_X, test_data_y)

        # Model size
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

        result = {
            'model_name': model_name,
            'model_path': model_path,
            'model_type': 'character',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': roc_auc,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'model_size_mb': float(model_size_mb),
            'calibration': calibration,
            'cost_sensitive_metrics': cost_sensitive,
            'overfitting_indicators': overfitting,
            'status': 'SUCCESS'
        }

        return result

    except Exception as e:
        return {
            'model_name': model_name,
            'model_path': model_path,
            'model_type': 'character',
            'status': 'FAILED',
            'error': str(e)
        }


def evaluate_features_model(model_path, mlinfo_path, test_data, test_data_y, model_name):
    """Evaluate multi-input features model"""
    if not FEATURES_AVAILABLE:
        return {'model_name': model_name, 'status': 'SKIPPED', 'reason': 'feature_utils not available'}

    try:
        from tensorflow.keras.models import Model
        model = load_model_with_custom_objects(model_path, model_name)

        # Load vocabulary
        with open(mlinfo_path, 'r') as f:
            mlinfo = json.load(f)

        char_list = mlinfo['vocabulary']['char_list']
        char_indices = dict((c, i+1) for i, c in enumerate(char_list))

        # Encode characters
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
        print(f"  Extracting features for {len(test_data)} samples...")
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
        y_pred_proba = model.predict([X_char, X_features], batch_size=64, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Calculate standard metrics
        accuracy = accuracy_score(test_data_y, y_pred)
        precision = precision_score(test_data_y, y_pred, zero_division=0)
        recall = recall_score(test_data_y, y_pred, zero_division=0)
        f1 = f1_score(test_data_y, y_pred, zero_division=0)

        cm = confusion_matrix(test_data_y, y_pred)
        tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0, 0, 0, 0)

        # Calculate additional metrics
        roc_auc = calculate_roc_auc(test_data_y, y_pred_proba)
        calibration = calculate_calibration_metrics(test_data_y, y_pred_proba)
        cost_sensitive = calculate_cost_sensitive_metrics(test_data_y, y_pred)
        overfitting = calculate_overfitting_metrics(model, [X_char, X_features], test_data_y)

        # Model size
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

        result = {
            'model_name': model_name,
            'model_path': model_path,
            'model_type': 'features',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': roc_auc,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'model_size_mb': float(model_size_mb),
            'features_count': 21,
            'calibration': calibration,
            'cost_sensitive_metrics': cost_sensitive,
            'overfitting_indicators': overfitting,
            'status': 'SUCCESS'
        }

        return result

    except Exception as e:
        return {
            'model_name': model_name,
            'model_path': model_path,
            'model_type': 'features',
            'status': 'FAILED',
            'error': str(e)
        }


def evaluate_xgboost_model(model_path, mlinfo_path, test_data, test_data_y, model_name):
    """Evaluate XGBoost model"""
    if not FEATURES_AVAILABLE:
        return {'model_name': model_name, 'status': 'SKIPPED', 'reason': 'feature_utils not available'}

    try:
        import pickle
        try:
            import xgboost as xgb
        except ImportError:
            return {'model_name': model_name, 'status': 'SKIPPED', 'reason': 'xgboost not installed'}

        # Load XGBoost model
        with open(model_path, 'rb') as f:
            xgb_model = pickle.load(f)

        # Extract engineered features for test data
        print(f"  Extracting engineered features for test set...")
        X_test_features = []
        for idx, (label, package_name) in enumerate(test_data):
            if idx % 5000 == 0:
                print(f"    Processed {idx}/{len(test_data)}")
            features, _ = extract_features_to_array(package_name)
            X_test_features.append(features)

        X_test_features = np.array(X_test_features, dtype=np.float32)

        # Get predictions
        dtest = xgb.DMatrix(X_test_features)
        y_pred_proba = xgb_model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(test_data_y, y_pred)
        precision = precision_score(test_data_y, y_pred, zero_division=0)
        recall = recall_score(test_data_y, y_pred, zero_division=0)
        f1 = f1_score(test_data_y, y_pred, zero_division=0)

        cm = confusion_matrix(test_data_y, y_pred)
        tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0, 0, 0, 0)

        # Calculate additional metrics
        roc_auc = calculate_roc_auc(test_data_y, y_pred_proba)
        calibration = calculate_calibration_metrics(test_data_y, y_pred_proba)
        cost_sensitive = calculate_cost_sensitive_metrics(test_data_y, y_pred)

        # Model size
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

        result = {
            'model_name': model_name,
            'model_path': model_path,
            'model_type': 'xgboost',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': roc_auc,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'model_size_mb': float(model_size_mb),
            'calibration': calibration,
            'cost_sensitive_metrics': cost_sensitive,
            'status': 'SUCCESS'
        }

        return result

    except Exception as e:
        return {
            'model_name': model_name,
            'model_path': model_path,
            'model_type': 'xgboost',
            'status': 'FAILED',
            'error': str(e)
        }


def evaluate_stacked_model(lstm_model_path, xgb_model_path, mlinfo_path, test_data_X_char, test_data_X_features, test_data_y, model_name):
    """Evaluate stacked XGBoost+LSTM model"""
    try:
        from tensorflow.keras.models import Model

        # Load LSTM model with custom objects support
        lstm_model = load_model_with_custom_objects(lstm_model_path, model_name)

        # Load mlinfo to get vocabulary
        with open(mlinfo_path, 'r') as f:
            mlinfo = json.load(f)

        # Create feature extractor (remove final dense layer)
        feature_extractor = Model(
            inputs=lstm_model.input,
            outputs=lstm_model.layers[-2].output
        )

        # Extract LSTM features
        print(f"  Extracting LSTM features from test set...")
        X_lstm_test = feature_extractor.predict(test_data_X_char, batch_size=64, verbose=0)

        # Concatenate LSTM features with engineered features
        X_combined_test = np.concatenate([X_lstm_test, test_data_X_features], axis=1)

        # Load XGBoost model
        with open(xgb_model_path, 'rb') as f:
            xgb_model = pickle.load(f)

        # Get predictions
        dtest = xgb.DMatrix(X_combined_test)
        y_pred_proba = xgb_model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(test_data_y, y_pred)
        precision = precision_score(test_data_y, y_pred, zero_division=0)
        recall = recall_score(test_data_y, y_pred, zero_division=0)
        f1 = f1_score(test_data_y, y_pred, zero_division=0)
        roc_auc = roc_auc_score(test_data_y, y_pred_proba)

        cm = confusion_matrix(test_data_y, y_pred)
        tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0, 0, 0, 0)

        # Model sizes
        lstm_size_mb = os.path.getsize(lstm_model_path) / (1024 * 1024)
        xgb_size_mb = os.path.getsize(xgb_model_path) / (1024 * 1024)
        total_size_mb = lstm_size_mb + xgb_size_mb

        result = {
            'model_name': model_name,
            'model_type': 'stacked',
            'lstm_model': lstm_model_path,
            'xgb_model': xgb_model_path,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': roc_auc,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'lstm_size_mb': float(lstm_size_mb),
            'xgb_size_mb': float(xgb_size_mb),
            'model_size_mb': float(total_size_mb),
            'status': 'SUCCESS'
        }

        return result

    except Exception as e:
        return {
            'model_name': model_name,
            'model_type': 'stacked',
            'status': 'FAILED',
            'error': str(e)
        }


def find_all_models():
    """Find all trained models including features model and stacked model"""
    models = []

    # Search for trained architecture models
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

    # Search for features model (in output/features/ directory or root)
    features_pattern = os.path.join(OUTPUT_DIR, 'features', 'features_model_*.keras')
    matching_features = glob.glob(features_pattern)
    if not matching_features:
        features_pattern = os.path.join(OUTPUT_DIR, 'features', 'features_model_*.hdf5')
        matching_features = glob.glob(features_pattern)
    if not matching_features:
        features_pattern = 'model_with_features_*.hdf5'
        matching_features = glob.glob(features_pattern)

    if matching_features:
        model_file = matching_features[-1]
        # Look for mlinfo in same directory or root
        mlinfo_pattern = os.path.join(OUTPUT_DIR, 'features', 'features_mlinfo_*.json')
        mlinfo_features = glob.glob(mlinfo_pattern)
        if not mlinfo_features:
            mlinfo_features = glob.glob('mlinfo_with_features_*.json')

        if mlinfo_features:
            mlinfo_file = mlinfo_features[-1]
            models.append({
                'arch': 'features',
                'model_file': model_file,
                'mlinfo_file': mlinfo_file,
                'model_type': 'features'
            })

    # Search for XGBoost model
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

    # Search for stacked model (XGBoost + LSTM)
    stacked_pattern = os.path.join(OUTPUT_DIR, 'xgboost_lstm_stacked', 'lstm_extractor_*.hdf5')
    matching_stacked_lstm = glob.glob(stacked_pattern)
    if matching_stacked_lstm:
        lstm_file = matching_stacked_lstm[-1]
        xgb_pattern = os.path.join(OUTPUT_DIR, 'xgboost_lstm_stacked', 'xgboost_stacked_*.pkl')
        matching_stacked_xgb = glob.glob(xgb_pattern)
        if matching_stacked_xgb:
            xgb_file = matching_stacked_xgb[-1]
            mlinfo_pattern = os.path.join(OUTPUT_DIR, 'xgboost_lstm_stacked', 'stacked_mlinfo_*.json')
            mlinfo_files = glob.glob(mlinfo_pattern)
            if mlinfo_files:
                mlinfo_file = mlinfo_files[-1]
                models.append({
                    'arch': 'stacked',
                    'lstm_file': lstm_file,
                    'xgb_file': xgb_file,
                    'mlinfo_file': mlinfo_file,
                    'model_type': 'stacked'
                })

    return models


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    print("=" * 80)
    print("ENHANCED MODEL EVALUATION AND COMPARISON")
    print("Includes: LSTM, BiLSTM, Transformer, CNN, CNN+LSTM, Features, XGBoost, and Stacked Models")
    print("=" * 80)
    print()

    # Load test data
    test_data = load_test_data()
    print()

    # Find all models
    models = find_all_models()

    if not models:
        print("ERROR: No trained models found!")
        print("Expected models in: models/output/{bilstm,transformer,cnn,cnn_lstm,dummy}/")
        print("Or features model: model_with_features_*.hdf5")
        return

    print(f"Found {len(models)} models to evaluate:\n")
    for m in models:
        if m['model_type'] == 'stacked':
            print(f"  ✓ {m['arch']}: LSTM + XGBoost")
        else:
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

    # Extract engineered features for stacked model (if needed)
    X_features_test = None
    if any(m['model_type'] == 'stacked' for m in models) and FEATURES_AVAILABLE:
        print("Extracting engineered features for stacked model...")
        X_features_test = []
        for idx, (label, package_name) in enumerate(test_data):
            if idx % 5000 == 0:
                print(f"  Processed {idx}/{len(test_data)}")
            features, _ = extract_features_to_array(package_name)
            X_features_test.append(features)
        X_features_test = np.array(X_features_test, dtype=np.float32)

        # Normalize features
        scaler = MinMaxScaler()
        X_features_test = scaler.fit_transform(X_features_test)
        print(f"Engineered features shape: {X_features_test.shape}\n")

    # Evaluate all models
    print("Evaluating models...\n")
    results = []

    for model_info in models:
        arch = model_info['arch']
        print(f"Evaluating {arch}...", end=' ')

        if model_info['model_type'] == 'character':
            result = evaluate_character_model(
                model_info['model_file'],
                model_info['mlinfo_file'],
                X_test,
                y_test,
                arch
            )
        elif model_info['model_type'] == 'features':
            result = evaluate_features_model(
                model_info['model_file'],
                model_info['mlinfo_file'],
                test_data,
                y_test,
                arch
            )
        elif model_info['model_type'] == 'xgboost':
            result = evaluate_xgboost_model(
                model_info['model_file'],
                model_info['mlinfo_file'],
                test_data,
                y_test,
                arch
            )
        elif model_info['model_type'] == 'stacked':
            result = evaluate_stacked_model(
                model_info['lstm_file'],
                model_info['xgb_file'],
                model_info['mlinfo_file'],
                X_test,
                X_features_test,
                y_test,
                arch
            )
        else:
            result = {'status': 'SKIPPED', 'reason': f'Unknown model type: {model_info["model_type"]}'}

        if result['status'] == 'SUCCESS':
            print(f"✓ Accuracy: {result['accuracy']:.4f}")
            results.append(result)
        elif result['status'] == 'SKIPPED':
            print(f"⊘ Skipped: {result.get('reason', 'Unknown')}")
        else:
            print(f"✗ Failed: {result.get('error', 'Unknown error')}")

    print()
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print()

    # Sort by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)

    # Print table
    print(f"{'Model':<15} {'Type':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Size (MB)':<10}")
    print("-" * 95)

    for result in results:
        model_type = result.get('model_type', 'unknown')
        print(f"{result['model_name']:<15} {model_type:<12} {result['accuracy']:.4f}      {result['precision']:.4f}       {result['recall']:.4f}       {result['f1_score']:.4f}      {result['model_size_mb']:.2f}")

    print()
    print("RECOMMENDATION:")
    print("-" * 80)

    if results:
        best = results[0]
        print(f"🏆 Best model: {best['model_name']}")
        print(f"   - Accuracy: {best['accuracy']:.4f}")
        print(f"   - Precision: {best['precision']:.4f}")
        print(f"   - Recall: {best['recall']:.4f}")
        print(f"   - F1-Score: {best['f1_score']:.4f}")
        if best.get('roc_auc'):
            print(f"   - ROC-AUC: {best['roc_auc']:.4f}")
        if best.get('calibration', {}).get('calibration_score'):
            print(f"   - Calibration Score: {best['calibration']['calibration_score']:.4f}")
        if best.get('cost_sensitive_metrics', {}).get('weighted_accuracy'):
            print(f"   - Cost-Sensitive Accuracy: {best['cost_sensitive_metrics']['weighted_accuracy']:.4f}")
        print(f"   - Model Size: {best['model_size_mb']:.2f} MB")
        print(f"   - Type: {best.get('model_type', 'unknown')}")
        print(f"   - Path: {best['model_path']}")
        print()

        # Show improvement over dummy model
        dummy = [r for r in results if 'dummy' in r['model_name'].lower()]
        if dummy:
            dummy_acc = dummy[0]['accuracy']
            improvement = (best['accuracy'] - dummy_acc) / dummy_acc * 100
            print(f"Improvement over dummy model: +{improvement:.1f}%")
        print()

    # Save detailed report
    report_file = f'model_comparison_all_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Detailed report saved to: {report_file}")
    print()

    # Generate markdown report
    md_report = f'model_comparison_all_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
    with open(md_report, 'w') as f:
        f.write("# Comprehensive Model Comparison Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        f.write("## Summary\n\n")
        f.write(f"Evaluated {len(results)} models on test set\n\n")

        f.write("## Performance Comparison\n\n")
        f.write("| Model | Type | Accuracy | Precision | Recall | F1 | ROC-AUC | Cal. Score | Size (MB) |\n")
        f.write("|-------|------|----------|-----------|--------|----|---------|-----------|-----------|\n")

        for result in results:
            model_type = result.get('model_type', 'unknown')
            roc_auc = f"{result.get('roc_auc', 0):.4f}" if result.get('roc_auc') else "N/A"
            cal_score = f"{result.get('calibration', {}).get('calibration_score', 0):.4f}" if result.get('calibration', {}).get('calibration_score') else "N/A"
            f.write(f"| {result['model_name']} | {model_type} | {result['accuracy']:.4f} | {result['precision']:.4f} | {result['recall']:.4f} | {result['f1_score']:.4f} | {roc_auc} | {cal_score} | {result['model_size_mb']:.2f} |\n")

        # Detailed results for each model
        f.write("\n## Detailed Results\n\n")
        for result in results:
            f.write(f"### {result['model_name']}\n\n")
            f.write(f"**Type:** {result.get('model_type', 'unknown')}\n\n")

            f.write("#### Classification Metrics\n")
            f.write(f"- **Accuracy:** {result['accuracy']:.4f}\n")
            f.write(f"- **Precision:** {result['precision']:.4f}\n")
            f.write(f"- **Recall:** {result['recall']:.4f}\n")
            f.write(f"- **F1-Score:** {result['f1_score']:.4f}\n\n")

            if result.get('roc_auc'):
                f.write("#### Advanced Metrics\n")
                f.write(f"- **ROC-AUC:** {result['roc_auc']:.4f}\n")

            if result.get('calibration'):
                calibration = result['calibration']
                f.write(f"- **Mean Confidence:** {calibration.get('mean_confidence', 'N/A'):.4f}\n")
                f.write(f"- **Calibration Error:** {calibration.get('calibration_error', 'N/A'):.4f}\n")
                f.write(f"- **Calibration Score:** {calibration.get('calibration_score', 'N/A'):.4f}\n\n")

            if result.get('cost_sensitive_metrics'):
                cost = result['cost_sensitive_metrics']
                f.write("#### Cost-Sensitive Metrics (FN cost = 2.0x, FP cost = 1.0x)\n")
                f.write(f"- **Total Cost:** {cost.get('total_cost', 'N/A')}\n")
                f.write(f"- **Cost per Sample:** {cost.get('cost_per_sample', 'N/A'):.4f}\n")
                f.write(f"- **Weighted Accuracy:** {cost.get('weighted_accuracy', 'N/A'):.4f}\n\n")

            if result.get('overfitting_indicators'):
                overfit = result['overfitting_indicators']
                f.write("#### Overfitting Indicators\n")
                f.write(f"- **Overconfidence Ratio:** {overfit.get('overconfidence_ratio', 'N/A'):.4f}\n")
                f.write(f"- **Mean Prediction Margin:** {overfit.get('mean_prediction_margin', 'N/A'):.4f}\n\n")

            f.write(f"#### Confusion Matrix\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| True Negatives | {result['true_negatives']} |\n")
            f.write(f"| False Positives | {result['false_positives']} |\n")
            f.write(f"| False Negatives | {result['false_negatives']} |\n")
            f.write(f"| True Positives | {result['true_positives']} |\n\n")

            f.write(f"**Model Size:** {result['model_size_mb']:.2f} MB\n\n")
            f.write("---\n\n")

        if results:
            best = results[0]
            f.write("## Best Model\n\n")
            f.write(f"**{best['model_name']}** (Type: {best.get('model_type', 'unknown')})\n\n")
            f.write(f"- Accuracy: {best['accuracy']:.2%}\n")
            f.write(f"- Precision: {best['precision']:.4f}\n")
            f.write(f"- Recall: {best['recall']:.4f}\n")
            f.write(f"- F1-Score: {best['f1_score']:.4f}\n")
            if best.get('roc_auc'):
                f.write(f"- ROC-AUC: {best['roc_auc']:.4f}\n")
            f.write(f"- Model Size: {best['model_size_mb']:.2f} MB\n\n")

            f.write("## Deployment\n\n")
            f.write(f"```bash\n")
            f.write(f"python deploy.py --model {best['model_path']} \\\n")
            f.write(f"                 --mlinfo {best.get('mlinfo_file', 'mlinfo.json')}\n")
            f.write(f"```\n")

    print(f"Markdown report saved to: {md_report}")
    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
