#!/usr/bin/env python3
"""
Generate evaluation summary report from Docker evaluation results

Usage:
    python3 generate_evaluation_summary.py <results_directory>
"""

import os
import json
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

def load_results(results_dir):
    """Load all evaluation results"""
    results = []
    result_files = Path(results_dir).glob('*_evaluation_result.json')

    for result_file in sorted(result_files):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Error loading {result_file}: {e}")

    return results


def generate_markdown_report(results, output_file):
    """Generate markdown report"""
    if not results:
        return

    successful = [r for r in results if r.get('status') == 'SUCCESS']
    failed = [r for r in results if r.get('status') == 'FAILED']

    with open(output_file, 'w') as f:
        f.write("# Model Evaluation Summary Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write(f"- Total Models: {len(results)}\n")
        f.write(f"- Successful: {len(successful)}\n")
        f.write(f"- Failed: {len(failed)}\n\n")

        # Successful models table
        if successful:
            f.write("## Successfully Evaluated Models\n\n")
            f.write("| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Size (MB) |\n")
            f.write("|-------|----------|-----------|--------|----------|---------|----------|\n")

            for result in sorted(successful, key=lambda x: x.get('accuracy', 0), reverse=True):
                f.write(f"| {result.get('model_name')} ")
                f.write(f"| {result.get('accuracy', 0):.4f} ")
                f.write(f"| {result.get('precision', 0):.4f} ")
                f.write(f"| {result.get('recall', 0):.4f} ")
                f.write(f"| {result.get('f1_score', 0):.4f} ")
                f.write(f"| {result.get('roc_auc', 0):.4f} ")
                f.write(f"| {result.get('model_size_mb', 0):.2f} ")
                f.write("|\n")

            f.write("\n")

        # Failed models
        if failed:
            f.write("## Failed Models\n\n")
            for result in failed:
                f.write(f"- **{result.get('model_name')}**: {result.get('error', 'Unknown error')}\n")

            f.write("\n")

        # Detailed results
        f.write("## Detailed Results\n\n")
        for result in sorted(successful, key=lambda x: x.get('accuracy', 0), reverse=True):
            f.write(f"### {result.get('model_name')}\n\n")
            f.write(f"**Status:** SUCCESS\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Accuracy | {result.get('accuracy', 0):.4f} |\n")
            f.write(f"| Precision | {result.get('precision', 0):.4f} |\n")
            f.write(f"| Recall | {result.get('recall', 0):.4f} |\n")
            f.write(f"| F1-Score | {result.get('f1_score', 0):.4f} |\n")
            f.write(f"| ROC-AUC | {result.get('roc_auc', 0):.4f} |\n")
            f.write(f"| True Negatives | {result.get('true_negatives', 0)} |\n")
            f.write(f"| False Positives | {result.get('false_positives', 0)} |\n")
            f.write(f"| False Negatives | {result.get('false_negatives', 0)} |\n")
            f.write(f"| True Positives | {result.get('true_positives', 0)} |\n")
            f.write(f"| Model Size | {result.get('model_size_mb', 0):.2f} MB |\n")
            f.write(f"| Timestamp | {result.get('timestamp', 'N/A')} |\n")
            f.write("\n")


def generate_json_report(results, output_file):
    """Generate JSON report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_models': len(results),
        'successful_count': sum(1 for r in results if r.get('status') == 'SUCCESS'),
        'failed_count': sum(1 for r in results if r.get('status') == 'FAILED'),
        'models': results
    }

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 generate_evaluation_summary.py <results_directory>")
        sys.exit(1)

    results_dir = sys.argv[1]

    if not os.path.isdir(results_dir):
        print(f"Error: {results_dir} is not a directory")
        sys.exit(1)

    print(f"Loading evaluation results from {results_dir}...")
    results = load_results(results_dir)

    if not results:
        print("No evaluation results found")
        return

    print(f"Found {len(results)} evaluation results")

    # Generate reports
    md_report = os.path.join(results_dir, 'EVALUATION_SUMMARY.md')
    json_report = os.path.join(results_dir, 'evaluation_summary.json')

    print(f"Generating markdown report: {md_report}")
    generate_markdown_report(results, md_report)

    print(f"Generating JSON report: {json_report}")
    generate_json_report(results, json_report)

    # Print summary to console
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    successful = [r for r in results if r.get('status') == 'SUCCESS']
    failed = [r for r in results if r.get('status') == 'FAILED']

    print(f"\nTotal Models: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}\n")

    if successful:
        print("Successfully Evaluated Models:")
        print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}")
        print("-" * 80)

        for result in sorted(successful, key=lambda x: x.get('accuracy', 0), reverse=True):
            print(f"{result.get('model_name', 'N/A'):<20} "
                  f"{result.get('accuracy', 0):<12.4f} "
                  f"{result.get('precision', 0):<12.4f} "
                  f"{result.get('recall', 0):<12.4f} "
                  f"{result.get('f1_score', 0):<12.4f} "
                  f"{result.get('roc_auc', 0):<12.4f}")

    if failed:
        print("\nFailed Models:")
        for result in failed:
            error = result.get('error', 'Unknown error')[:50]
            print(f"  ✗ {result.get('model_name')}: {error}")

    print("\n" + "="*80)
    print(f"Reports saved to: {results_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
