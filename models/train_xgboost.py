#!/usr/bin/env python3
"""
XGBoost training script for Android package name classification.

Demonstrates gradient boosting approach using engineered features
alongside the deep learning models (LSTM, Transformer, CNN, CNN+LSTM).
"""

import os
import sys
import csv
import codecs
import json
import time
import argparse
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

# Import feature extraction utilities
from feature_utils import extract_features_to_array


def load_data(label_0_files, label_1_files):
    """Load package names with labels from CSV files."""
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


def extract_xgboost_features(package_names):
    """Extract all features for XGBoost model using feature_utils.

    Uses the engineered features as arrays for direct XGBoost input.
    Returns both features and feature names for consistency.
    """
    feature_list = []
    feature_names = None

    for pname in package_names:
        # Extract features as array (returns tuple of (array, names))
        features, feature_names = extract_features_to_array(pname, feature_names)
        feature_list.append(features)

    return np.array(feature_list, dtype=np.float32), feature_names


def train_xgboost_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=1024):
    """Train XGBoost classifier.

    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (n_samples,)
        X_val: Validation features (n_samples, n_features)
        y_val: Validation labels (n_samples,)
        epochs: Number of boosting rounds
        batch_size: Batch size for training

    Returns:
        Trained XGBoost classifier
    """
    # Create DMatrix for efficient data loading
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 7,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1.0,
        'tree_method': 'hist',
        'device': 'cpu'
    }

    # Train with early stopping
    evals = [(dtrain, 'train'), (dval, 'validation')]
    evals_result = {}

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=epochs,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=8,
        verbose_eval=10
    )

    return model, evals_result


def evaluate_model(model, X_test, y_test, model_name="XGBoost"):
    """Evaluate model performance."""
    # Get predictions
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n{model_name} Performance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    print(f"    FN: {cm[1, 0]}, TP: {cm[1, 1]}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist()
    }


def save_model(model, output_dir, timestamp, metrics=None, evals_result=None, feature_importance=None, training_time=None):
    """Save XGBoost model, metadata, and performance metrics.

    Args:
        model: Trained XGBoost model
        output_dir: Output directory path
        timestamp: Timestamp string
        metrics: Dictionary of test metrics (accuracy, precision, recall, f1, roc_auc)
        evals_result: Training history from XGBoost (train/validation metrics per round)
        feature_importance: Dictionary of feature importance scores
        training_time: Training time in seconds
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, f"xgboost_model_{timestamp}.pkl")
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Model saved: {model_path}")
        print(f"  File exists: {os.path.exists(model_path)}")
    except Exception as e:
        print(f"⚠️ Error saving model: {e}")
        print(f"  File exists: {os.path.exists(model_path)}")

    # Save mlinfo with performance metrics
    mlinfo = {
        "version": int(datetime.now().strftime("%Y%m%d%H%M%S")),
        "model": f"xgboost_model_{timestamp}.pkl",
        "model_type": "xgboost",
        "features": {
            "type": "engineered",
            "count": 26,
            "description": "5 character-level features + 21 engineered features"
        },
        "training_date": datetime.now().isoformat(),
        "training_time_seconds": training_time if training_time else None,
        "test_metrics": metrics if metrics else {}
    }

    mlinfo_path = os.path.join(output_dir, f"xgboost_mlinfo_{timestamp}.json")
    try:
        with open(mlinfo_path, 'w') as f:
            json.dump(mlinfo, f, indent=2)
        print(f"✓ Metadata saved: {mlinfo_path}")
        print(f"  File exists: {os.path.exists(mlinfo_path)}")
    except Exception as e:
        print(f"⚠️ Error saving metadata: {e}")
        print(f"  File exists: {os.path.exists(mlinfo_path)}")

    # Save detailed training results (new file)
    training_results = {
        "model_name": "xgboost",
        "model_path": model_path,
        "timestamp": timestamp,
        "training_date": datetime.now().isoformat(),
        "training_time_seconds": training_time,
        "test_metrics": {
            "accuracy": float(metrics.get('accuracy', 0)) if metrics else 0,
            "precision": float(metrics.get('precision', 0)) if metrics else 0,
            "recall": float(metrics.get('recall', 0)) if metrics else 0,
            "f1_score": float(metrics.get('f1_score', 0)) if metrics else 0,
            "roc_auc": float(metrics.get('roc_auc', 0)) if metrics else 0
        },
        "confusion_matrix": metrics.get('confusion_matrix', {}) if metrics else {},
        "training_history": {
            "train": {
                "logloss": evals_result.get('train', {}).get('logloss', []) if evals_result else []
            },
            "validation": {
                "logloss": evals_result.get('validation', {}).get('logloss', []) if evals_result else []
            }
        } if evals_result else {},
        "feature_importance": feature_importance if feature_importance else {},
        "model_parameters": {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 7,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "early_stopping_rounds": 8
        }
    }

    results_path = os.path.join(output_dir, f"xgboost_training_results_{timestamp}.json")
    try:
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        print(f"✓ Training results saved: {results_path}")
        print(f"  File exists: {os.path.exists(results_path)}")
    except Exception as e:
        print(f"⚠️ Error saving training results: {e}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train XGBoost model for package name classification'
    )
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of boosting rounds')
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size for training')
    parser.add_argument('--patience', type=int, default=8,
                       help='Early stopping patience')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("XGBoost Training - Android Package Name Classifier")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Boosting Rounds: {args.epochs}")
    print(f"  Early Stopping Patience: {args.patience}")
    print(f"  Model Type: XGBoost (Gradient Boosting)")
    print(f"  Features: 26 (5 character-level + 21 engineered)")

    # Load data
    print(f"\nLoading training data...")
    start_time = time.time()

    data_dir = '/workspace/data'

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

    train_packages = [pkg for _, pkg in train_data]
    train_labels = np.array([label for label, _ in train_data])

    val_packages = [pkg for _, pkg in val_data]
    val_labels = np.array([label for label, _ in val_data])

    test_packages = [pkg for _, pkg in test_data]
    test_labels = np.array([label for label, _ in test_data])

    # Count by label
    train_packages_0 = sum(1 for label, _ in train_data if label == 0)
    train_packages_1 = sum(1 for label, _ in train_data if label == 1)

    val_packages_0 = sum(1 for label, _ in val_data if label == 0)
    val_packages_1 = sum(1 for label, _ in val_data if label == 1)

    test_packages_0 = sum(1 for label, _ in test_data if label == 0)
    test_packages_1 = sum(1 for label, _ in test_data if label == 1)

    print(f"  Train: {len(train_packages):,} samples ({train_packages_0:,} legitimate, {train_packages_1:,} suspicious)")
    print(f"  Validation: {len(val_packages):,} samples ({val_packages_0:,} legitimate, {val_packages_1:,} suspicious)")
    print(f"  Test: {len(test_packages):,} samples ({test_packages_0:,} legitimate, {test_packages_1:,} suspicious)")

    # Extract features
    print(f"\nExtracting features...")
    X_train, feature_names = extract_xgboost_features(train_packages)
    X_val, _ = extract_xgboost_features(val_packages)
    X_test, _ = extract_xgboost_features(test_packages)

    print(f"  Feature shape: {X_train.shape}")
    print(f"  Features extracted in {time.time() - start_time:.2f}s")

    # Train model
    print(f"\nTraining XGBoost model...")
    train_start = time.time()

    model, evals_result = train_xgboost_model(
        X_train, train_labels,
        X_val, val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    training_time = time.time() - train_start
    print(f"\nTraining completed in {training_time:.2f}s")

    # Evaluate
    print(f"\nEvaluating on test set...")
    metrics = evaluate_model(model, X_test, test_labels, "XGBoost")

    # Feature importance
    print(f"\nTop 10 Most Important Features:")
    importance = model.get_score(importance_type='weight')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for feat, score in sorted_importance:
        print(f"  {feat}: {score}")

    # Save model with complete metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.environ.get('OUTPUT_DIR', 'output/xgboost')

    # Convert importance dict to float values for JSON serialization
    importance_dict = {str(k): float(v) for k, v in importance.items()}

    save_model(
        model,
        output_dir,
        timestamp,
        metrics=metrics,
        evals_result=evals_result,
        feature_importance=importance_dict,
        training_time=training_time
    )

    # Summary
    print(f"\n" + "="*70)
    print("Training Summary:")
    print(f"  Model Type: XGBoost")
    print(f"  Training Time: {training_time:.2f}s")
    print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  Output Directory: {output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
