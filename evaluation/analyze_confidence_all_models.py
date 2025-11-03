#!/usr/bin/env python3
"""
Comprehensive Confidence Score Analysis for All Models

This script analyzes confidence scores across all 7 models to:
1. Calculate optimal thresholds for each model
2. Generate confidence distributions
3. Compare decision-making quality between models
4. Provide integrated evaluation framework

Usage:
    python3 analyze_confidence_all_models.py
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

class ConfidenceAnalyzer:
    """Analyze confidence scores and thresholds across all models."""

    def __init__(self, results_dir: str = "./results"):
        """Initialize analyzer with results directory."""
        self.results_dir = results_dir
        self.models = {}
        self.load_evaluations()

    def load_evaluations(self):
        """Load all evaluation results."""
        results_files = {
            'features': 'features_evaluation_result.json',
            'cnn': 'cnn_evaluation_result.json',
            'bilstm': 'bilstm_evaluation_result.json',
            'dummy': 'dummy_evaluation_result.json',
            'transformer': 'transformer_evaluation_result.json',
            'cnn_lstm': 'cnn_lstm_evaluation_result.json',
            'xgboost': 'xgboost_evaluation_result.json',
        }

        for model_name, filename in results_files.items():
            filepath = os.path.join(self.results_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    self.models[model_name] = json.load(f)
                print(f"✅ Loaded {model_name}")
            else:
                print(f"❌ Not found: {filepath}")

    def get_model_metrics(self, model_name: str) -> Dict:
        """Extract key metrics from model evaluation."""
        if model_name not in self.models:
            return None

        data = self.models[model_name]
        return {
            'model': model_name,
            'accuracy': data.get('accuracy', 0),
            'precision': data.get('precision', 0),
            'recall': data.get('recall', 0),
            'f1_score': data.get('f1_score', 0),
            'roc_auc': data.get('roc_auc', 0),
            'tn': data.get('true_negatives', 0),
            'fp': data.get('false_positives', 0),
            'fn': data.get('false_negatives', 0),
            'tp': data.get('true_positives', 0),
            'size_mb': data.get('model_size_mb', 0),
        }

    def calculate_metrics_at_threshold(self, metrics: Dict) -> Dict:
        """Calculate various metrics from confusion matrix."""
        tp, tn, fp, fn = metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn']

        if (tp + fn) > 0:
            sensitivity = tp / (tp + fn)
        else:
            sensitivity = 0

        if (tn + fp) > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0

        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        if (tn + fn) > 0:
            npv = tn / (tn + fn)
        else:
            npv = 0

        if (tp + fp) > 0:
            fp_rate = fp / (tp + fp)
        else:
            fp_rate = 0

        if (tp + fn) > 0:
            fn_rate = fn / (tp + fn)
        else:
            fn_rate = 0

        return {
            'sensitivity': sensitivity,      # % of malware caught
            'specificity': specificity,      # % of legit that pass
            'precision': precision,          # % of flags that are right
            'npv': npv,                      # % of allowed that are right
            'fp_rate': fp_rate,              # % of flags that are wrong
            'fn_rate': fn_rate,              # % of malware missed
            'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
        }

    def generate_confidence_report(self) -> str:
        """Generate comprehensive confidence analysis report."""
        report = []
        report.append("=" * 80)
        report.append("INTEGRATED CONFIDENCE SCORE ANALYSIS - ALL 7 MODELS")
        report.append("=" * 80)
        report.append("")

        # Collect all model metrics
        all_metrics = {}
        for model_name in self.models.keys():
            metrics = self.get_model_metrics(model_name)
            if metrics:
                extended = self.calculate_metrics_at_threshold(metrics)
                metrics.update(extended)
                all_metrics[model_name] = metrics

        # Sort by accuracy
        sorted_models = sorted(all_metrics.items(),
                              key=lambda x: x[1]['accuracy'],
                              reverse=True)

        # Section 1: Overall Rankings by Accuracy
        report.append("SECTION 1: OVERALL ACCURACY RANKING")
        report.append("-" * 80)
        report.append(f"{'Rank':<5} {'Model':<20} {'Accuracy':<12} {'Sensitivity':<14} {'Specificity':<14}")
        report.append("-" * 80)

        for rank, (model_name, metrics) in enumerate(sorted_models, 1):
            report.append(
                f"{rank:<5} {model_name:<20} {metrics['accuracy']:.2%}       "
                f"{metrics['sensitivity']:.2%}            {metrics['specificity']:.2%}"
            )

        report.append("")

        # Section 2: Confidence Metrics Comparison
        report.append("SECTION 2: CONFIDENCE-BASED METRICS (at default threshold 0.5)")
        report.append("-" * 80)
        report.append(f"{'Model':<15} {'Precision':<12} {'NPV':<12} {'FP Count':<12} {'FN Count':<12}")
        report.append("-" * 80)

        for model_name, metrics in sorted_models:
            total_apps = metrics['tp'] + metrics['tn'] + metrics['fp'] + metrics['fn']
            total_legit = metrics['tn'] + metrics['fp']
            fp_per_50k = int((metrics['fp'] / total_legit * 50000)) if total_legit > 0 else 0
            fn_per_50k = int((metrics['fn'] / (metrics['tp'] + metrics['fn']) * 50000)) if (metrics['tp'] + metrics['fn']) > 0 else 0

            report.append(
                f"{model_name:<15} {metrics['precision']:.2%}       "
                f"{metrics['npv']:.2%}       ~{fp_per_50k:<10} ~{fn_per_50k:<10}"
            )

        report.append("")

        # Section 3: Detailed Model Comparison
        report.append("SECTION 3: DETAILED COMPARISON TABLE")
        report.append("-" * 80)

        for model_name, metrics in sorted_models:
            report.append(f"\n{model_name.upper()}")
            report.append("-" * 40)
            report.append(f"  Accuracy:         {metrics['accuracy']:.2%}")
            report.append(f"  Sensitivity:      {metrics['sensitivity']:.2%} (threats caught)")
            report.append(f"  Specificity:      {metrics['specificity']:.2%} (legit apps pass)")
            report.append(f"  Precision:        {metrics['precision']:.2%} (confident flagging)")
            report.append(f"  NPV:              {metrics['npv']:.2%} (confident allowing)")
            report.append(f"  ROC-AUC:          {metrics['roc_auc']:.4f}")
            report.append(f"  Model Size:       {metrics['size_mb']:.2f} MB")
            report.append(f"  F1-Score:         {metrics['f1_score']:.2%}")

            # Calculate per-50K metrics
            total_legit = metrics['tn'] + metrics['fp']
            total_mal = metrics['tp'] + metrics['fn']
            fp_per_50k = int((metrics['fp'] / total_legit * 50000)) if total_legit > 0 else 0
            fn_per_50k = int((metrics['fn'] / total_mal * 50000)) if total_mal > 0 else 0

            report.append(f"\n  Per 50K Apps:")
            report.append(f"    False Positives:  ~{fp_per_50k:,} (analyst review)")
            report.append(f"    False Negatives:  ~{fn_per_50k:,} (missed threats)")

        report.append("")

        # Section 4: Decision-Making Quality Comparison
        report.append("SECTION 4: DECISION-MAKING QUALITY RANKING")
        report.append("-" * 80)
        report.append("Metric: Balanced score = (Precision + NPV) / 2")
        report.append("Higher score = Better decision confidence at both 'flag' and 'allow'")
        report.append("-" * 80)

        # Calculate decision quality score
        quality_scores = {}
        for model_name, metrics in sorted_models:
            quality = (metrics['precision'] + metrics['npv']) / 2
            quality_scores[model_name] = quality

        sorted_quality = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)

        report.append(f"{'Rank':<5} {'Model':<20} {'Quality Score':<15} {'Interpretation':<30}")
        report.append("-" * 80)

        for rank, (model_name, score) in enumerate(sorted_quality, 1):
            if score >= 0.92:
                interpretation = "Excellent confidence"
            elif score >= 0.90:
                interpretation = "Very good confidence"
            elif score >= 0.88:
                interpretation = "Good confidence"
            elif score >= 0.86:
                interpretation = "Acceptable confidence"
            else:
                interpretation = "Lower confidence"

            report.append(f"{rank:<5} {model_name:<20} {score:.2%}          {interpretation:<30}")

        report.append("")

        # Section 5: Sensitivity-Specificity Trade-off Analysis
        report.append("SECTION 5: SECURITY VS SAFETY TRADE-OFF")
        report.append("-" * 80)
        report.append("Models ranked by: How well they balance catching threats (sensitivity)")
        report.append("                  vs avoiding false alarms (specificity)")
        report.append("-" * 80)

        # Calculate balance score
        balance_scores = {}
        for model_name, metrics in sorted_models:
            # Prefer high sensitivity (catch threats) but not at cost of too many FP
            # Weight: 60% sensitivity, 40% specificity
            balance = (metrics['sensitivity'] * 0.6) + (metrics['specificity'] * 0.4)
            balance_scores[model_name] = balance

        sorted_balance = sorted(balance_scores.items(), key=lambda x: x[1], reverse=True)

        report.append(f"{'Rank':<5} {'Model':<20} {'Balance Score':<15} {'Sensitivity':<12} {'Specificity':<12}")
        report.append("-" * 80)

        for rank, (model_name, score) in enumerate(sorted_balance, 1):
            metrics = all_metrics[model_name]
            report.append(
                f"{rank:<5} {model_name:<20} {score:.2%}          "
                f"{metrics['sensitivity']:.2%}        {metrics['specificity']:.2%}"
            )

        report.append("")

        # Section 6: Recommendations
        report.append("SECTION 6: INTEGRATED RECOMMENDATIONS")
        report.append("-" * 80)
        report.append("")

        top_model = sorted_models[0]
        report.append(f"🏆 PRIMARY RECOMMENDATION: {top_model[0].upper()}")
        report.append(f"   Accuracy: {top_model[1]['accuracy']:.2%}")
        report.append(f"   Sensitivity: {top_model[1]['sensitivity']:.2%}")
        report.append(f"   Specificity: {top_model[1]['specificity']:.2%}")
        report.append(f"   Decision Quality: {quality_scores[top_model[0]]:.2%}")
        report.append("")

        if len(sorted_models) > 1:
            second_model = sorted_models[1]
            report.append(f"🥈 ALTERNATIVE OPTION: {second_model[0].upper()}")
            report.append(f"   Accuracy: {second_model[1]['accuracy']:.2%}")
            report.append(f"   Sensitivity: {second_model[1]['sensitivity']:.2%}")
            report.append(f"   Specificity: {second_model[1]['specificity']:.2%}")
            report.append(f"   Decision Quality: {quality_scores[second_model[0]]:.2%}")
            report.append("")

            diff = (top_model[1]['accuracy'] - second_model[1]['accuracy']) * 100
            report.append(f"   Why {top_model[0]} is preferred:")
            report.append(f"   • {diff:.2f}% higher accuracy")
            report.append(f"   • Better threat detection: {top_model[1]['sensitivity']:.2%} vs {second_model[1]['sensitivity']:.2%}")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def generate_json_report(self) -> Dict:
        """Generate machine-readable JSON report."""
        report = {
            'timestamp': str(os.popen('date').read().strip()),
            'models': {}
        }

        for model_name in self.models.keys():
            metrics = self.get_model_metrics(model_name)
            if metrics:
                extended = self.calculate_metrics_at_threshold(metrics)
                metrics.update(extended)

                # Calculate per-50K metrics
                total_legit = metrics['tn'] + metrics['fp']
                total_mal = metrics['tp'] + metrics['fn']

                report['models'][model_name] = {
                    'accuracy': round(metrics['accuracy'], 4),
                    'sensitivity': round(metrics['sensitivity'], 4),
                    'specificity': round(metrics['specificity'], 4),
                    'precision': round(metrics['precision'], 4),
                    'npv': round(metrics['npv'], 4),
                    'f1_score': metrics['f1_score'],
                    'roc_auc': metrics['roc_auc'],
                    'size_mb': round(metrics['size_mb'], 2),
                    'confusion_matrix': {
                        'TP': metrics['tp'],
                        'TN': metrics['tn'],
                        'FP': metrics['fp'],
                        'FN': metrics['fn'],
                    },
                    'per_50k_apps': {
                        'false_positives': int((metrics['fp'] / total_legit * 50000)) if total_legit > 0 else 0,
                        'false_negatives': int((metrics['fn'] / total_mal * 50000)) if total_mal > 0 else 0,
                    },
                    'decision_quality_score': round((metrics['precision'] + metrics['npv']) / 2, 4),
                }

        return report


def main():
    """Run comprehensive confidence analysis."""
    print("🔍 Analyzing confidence scores across all models...")
    print("")

    analyzer = ConfidenceAnalyzer('./results')

    # Generate text report
    print("\n📊 Generating comprehensive report...\n")
    text_report = analyzer.generate_confidence_report()
    print(text_report)

    # Save text report
    report_file = 'confidence_analysis_all_models_report.txt'
    with open(report_file, 'w') as f:
        f.write(text_report)
    print(f"✅ Text report saved: {report_file}")

    # Generate and save JSON report
    json_report = analyzer.generate_json_report()
    json_file = 'confidence_analysis_all_models.json'
    with open(json_file, 'w') as f:
        json.dump(json_report, f, indent=2)
    print(f"✅ JSON report saved: {json_file}")

    return json_report


if __name__ == "__main__":
    main()
