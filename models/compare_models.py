#!/usr/bin/env python3
"""
Model Comparison Script

Trains all model architectures and compares their performance on the same dataset.
Generates a comprehensive comparison report with metrics and timing information.

Available Models:
1. Dummy LSTM (baseline) - 2-layer LSTM for comparison
2. BiLSTM - Bidirectional LSTM, expected +2-3% accuracy
3. Transformer - Multi-head attention, expected +4-6% accuracy
4. CNN - 1D Convolutional, 10x faster inference
5. CNN+LSTM Hybrid - CNN features + LSTM sequences, expected +1-3% accuracy
6. Features Model (NEW!) - Multi-input with 21 engineered features, expected +5-7% accuracy
7. XGBoost (NEW!) - Gradient boosting with engineered features, expected +3-5% accuracy

Evaluation Criteria (Updated):
- Test Accuracy: Minimum 80% acceptable (excellent ≥85%)
- F1 Score: Minimum 78% (now 40% weight in scoring)
- Precision & Recall: 40% combined weight (prioritizes security metrics)
"""

import argparse
import subprocess
import json
import time
import os
from datetime import datetime
from pathlib import Path
import sys

# Model configurations
MODELS = {
    'baseline_lstm': {
        'script': 'train_dummy.py',
        'name': 'Baseline LSTM',
        'description': 'Original 2-layer LSTM model',
        'args': []
    },
    'bilstm': {
        'script': 'train_bilstm.py',
        'name': 'Bidirectional LSTM',
        'description': '2-layer BiLSTM model',
        'args': []
    },
    'transformer': {
        'script': 'train_transformer.py',
        'name': 'Transformer',
        'description': 'Multi-head attention with positional encoding',
        'args': []
    },
    'cnn': {
        'script': 'train_cnn.py',
        'name': '1D CNN',
        'description': 'Lightweight convolutional model',
        'args': []
    },
    'cnn_lstm': {
        'script': 'train_cnn_lstm.py',
        'name': 'CNN+LSTM Hybrid',
        'description': 'CNN for features, LSTM for sequences',
        'args': []
    },
    'features': {
        'script': 'train_with_features.py',
        'name': 'Features-Enhanced LSTM',
        'description': 'Multi-input LSTM with 21 engineered features (entropy, patterns, composition)',
        'args': ['--batch-size', '512']  # Features model uses smaller batch size
    },
    'xgboost': {
        'script': 'train_xgboost.py',
        'name': 'XGBoost',
        'description': 'Gradient boosting with engineered features (alternative to deep learning)',
        'args': []
    },
    'stacked': {
        'script': 'train_xgboost_lstm_stacked.py',
        'name': 'Stacked XGBoost+LSTM',
        'description': 'Ensemble approach: LSTM feature extractor + XGBoost meta-learner (expected +6-8%)',
        'args': []
    }
}


def run_training(model_key, config, common_args):
    """Run training for a single model"""

    print("\n" + "="*80)
    print(f"TRAINING: {config['name']}")
    print("="*80)
    print(f"Description: {config['description']}")
    print(f"Script: {config['script']}")
    print()

    script_path = Path(config['script'])
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return None

    # Determine output directory based on model
    output_dir = f"output/{model_key}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build command with OUTPUT_DIR environment variable
    cmd = ['python3', str(script_path)]
    cmd.extend(common_args)
    cmd.extend(config['args'])

    print(f"Command: OUTPUT_DIR={output_dir} {' '.join(cmd)}")
    print()

    start_time = time.time()

    try:
        # Run training with OUTPUT_DIR environment variable
        env = os.environ.copy()
        env['OUTPUT_DIR'] = output_dir

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=env
        )

        training_time = time.time() - start_time

        # Extract metrics from output
        output = result.stdout

        # Parse metrics from output
        metrics = extract_metrics(output)
        metrics['training_time'] = training_time
        metrics['model_key'] = model_key
        metrics['model_name'] = config['name']

        # Find generated mlinfo file in output directory
        # Search in: output/{model_key}/*mlinfo*.json
        mlinfo_files = list(Path(output_dir).glob('*mlinfo*.json'))

        # Fallback: search current directory for .keras files (TF 2.13+)
        if not mlinfo_files:
            mlinfo_files = list(Path(output_dir).glob('*mlinfo*.json'))

        if mlinfo_files:
            mlinfo_path = sorted(mlinfo_files)[-1]  # Get latest
            try:
                with open(mlinfo_path) as f:
                    mlinfo = json.load(f)
                    metrics['model_file'] = mlinfo.get('model', 'unknown')
                    metrics['mlinfo_file'] = str(mlinfo_path)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read mlinfo file {mlinfo_path}: {e}")

        print(f"\n✓ Training completed in {training_time:.1f}s")
        print(f"  Accuracy: {metrics.get('accuracy', 'N/A')}")
        print(f"  F1-Score: {metrics.get('f1', 'N/A')}")

        return metrics

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed:")
        print(e.stderr)
        return None


def extract_metrics(output):
    """Extract accuracy, precision, recall, F1 from training output"""

    metrics = {}

    lines = output.split('\n')
    for i, line in enumerate(lines):
        # Look for final metrics
        if 'Accuracy:' in line:
            try:
                metrics['accuracy'] = float(line.split(':')[1].strip())
            except:
                pass

        if 'Precision:' in line:
            try:
                metrics['precision'] = float(line.split(':')[1].strip())
            except:
                pass

        if 'Recall:' in line:
            try:
                metrics['recall'] = float(line.split(':')[1].strip())
            except:
                pass

        if 'F1-Score:' in line:
            try:
                metrics['f1'] = float(line.split(':')[1].strip())
            except:
                pass

    return metrics


def generate_report(results, output_file='comparison_report.md'):
    """Generate markdown comparison report"""

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    report = f"""# Model Comparison Report

**Generated:** {timestamp}

## Summary

This report compares {len(results)} different model architectures trained on the same Android package name dataset.

"""

    # Add comparison table
    report += """## Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
"""

    # Sort by F1 score (descending)
    sorted_results = sorted(results, key=lambda x: x.get('f1', 0), reverse=True)

    for r in sorted_results:
        model_name = r.get('model_name', 'Unknown')
        accuracy = f"{r.get('accuracy', 0):.4f}" if r.get('accuracy') else 'N/A'
        precision = f"{r.get('precision', 0):.4f}" if r.get('precision') else 'N/A'
        recall = f"{r.get('recall', 0):.4f}" if r.get('recall') else 'N/A'
        f1 = f"{r.get('f1', 0):.4f}" if r.get('f1') else 'N/A'
        train_time = f"{r.get('training_time', 0):.1f}s" if r.get('training_time') else 'N/A'

        report += f"| {model_name} | {accuracy} | {precision} | {recall} | {f1} | {train_time} |\n"

    # Add detailed results
    report += "\n## Detailed Results\n\n"

    for r in sorted_results:
        report += f"### {r.get('model_name', 'Unknown')}\n\n"
        report += f"- **Model File:** {r.get('model_file', 'N/A')}\n"
        report += f"- **Config File:** {r.get('mlinfo_file', 'N/A')}\n"
        report += f"- **Training Time:** {r.get('training_time', 0):.1f}s\n"
        report += f"- **Accuracy:** {r.get('accuracy', 'N/A')}\n"
        report += f"- **Precision:** {r.get('precision', 'N/A')}\n"
        report += f"- **Recall:** {r.get('recall', 'N/A')}\n"
        report += f"- **F1-Score:** {r.get('f1', 'N/A')}\n"
        report += "\n"

    # Add recommendations
    report += "## Recommendations\n\n"

    best_f1 = sorted_results[0] if sorted_results else None
    if best_f1:
        report += f"**Best Overall Performance:** {best_f1.get('model_name')} "
        report += f"(F1: {best_f1.get('f1', 0):.4f})\n\n"

    report += """
**Updated Selection Criteria (Oct 2025):**

**Minimum Production Requirements:**
- Test Accuracy: ≥ 80%
- F1 Score: ≥ 78%
- Precision (minimize false alarms): ≥ 75%
- Recall (catch threats): ≥ 80%

**Ideal Production Model:**
- Test Accuracy: ≥ 85%
- F1 Score: ≥ 82%
- Precision: ≥ 78%
- Recall: ≥ 82%

**Metric Weights:**
- Accuracy: 30% (down from 40%)
- F1 Score (Precision/Recall): 40% (up from 30%) ← PRIORITIZED FOR SECURITY
- Inference Speed: 20%
- Model Size: 10%

**Model Recommendations:**

1. **For Production (Highest Accuracy - Ensemble):**
   - ✅ **Stacked XGBoost+LSTM** (NEW - recommended, expected +6-8% improvement)
   - Combines LSTM feature extraction with XGBoost meta-learner
   - Best of both worlds: interpretability + accuracy
   - Target: F1 ≥ 0.84, Recall ≥ 0.84

2. **For Production (Security-Critical - Deep Learning):**
   - ✅ Features Model (recommended if ensemble unavailable, +5-7% improvement)
   - ✅ Transformer (if features model unavailable, +4-6% improvement)
   - Target: F1 ≥ 0.82, Recall ≥ 0.82

3. **For Production (Alternative Approach - Gradient Boosting):**
   - ✅ XGBoost (expected +3-5% improvement, explainable decisions, fast)
   - Similar performance to deep learning, faster training than deep learning
   - Target: F1 ≥ 0.78, Recall ≥ 0.80

4. **For Speed (High-Volume Requests >1000/sec):**
   - ✅ CNN model (10x faster inference than LSTM)
   - Acceptable if Accuracy ≥ 80%, F1 ≥ 0.78

5. **For Balance (Best accuracy/speed tradeoff):**
   - ✅ BiLSTM or CNN+LSTM Hybrid
   - Good recall without sacrificing speed too much

6. **Baseline Comparison:**
   - Dummy LSTM (benchmark only, not for production)

"""

    # Save report
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"\n{'='*80}")
    print(f"Report saved to: {output_file}")
    print(f"{'='*80}\n")

    # Also save JSON results
    json_file = output_file.replace('.md', '.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"JSON results saved to: {json_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare multiple model architectures on the same dataset'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        choices=list(MODELS.keys()) + ['all'],
        default=['all'],
        help='Models to train and compare (default: all)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1024,
        help='Batch size for training (default: 1024)'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=8,
        help='Early stopping patience (default: 8)'
    )

    parser.add_argument(
        '--output',
        default='comparison_report.md',
        help='Output report file (default: comparison_report.md)'
    )

    args = parser.parse_args()

    # Build common arguments
    common_args = [
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--patience', str(args.patience)
    ]

    # Determine which models to run
    models_to_run = args.models
    if 'all' in models_to_run:
        models_to_run = list(MODELS.keys())

    print("="*80)
    print("MODEL COMPARISON EXPERIMENT")
    print("="*80)
    print(f"\nModels to compare: {', '.join(models_to_run)}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Early stopping patience: {args.patience}")
    print()

    # Run all experiments
    results = []

    for model_key in models_to_run:
        if model_key not in MODELS:
            print(f"WARNING: Unknown model '{model_key}', skipping")
            continue

        config = MODELS[model_key]
        metrics = run_training(model_key, config, common_args)

        if metrics:
            results.append(metrics)

    # Generate report
    if results:
        generate_report(results, args.output)

        print("\n" + "="*80)
        print("COMPARISON COMPLETE")
        print("="*80)
        print(f"\nSuccessfully trained {len(results)}/{len(models_to_run)} models")
        print(f"Report: {args.output}")
    else:
        print("\n✗ No models completed successfully")
        sys.exit(1)


if __name__ == '__main__':
    main()
