#!/usr/bin/env python3
"""
Visualize Confidence Score Distributions for All Models

This script creates ASCII bar charts and formatted tables showing
confidence score distributions for all 7 models, separated by
legitimate (label 0) and malicious (label 1) apps.

Usage:
    python3 visualize_confidence_distributions.py
"""

import json
import sys

def generate_ascii_bar(percentage, max_width=50):
    """Generate an ASCII bar for given percentage"""
    filled_width = int((percentage / 100) * max_width)
    filled = '█' * filled_width
    empty = '░' * (max_width - filled_width)
    return filled + empty


def visualize_model_distribution(model_name, distribution):
    """Generate visualization for a single model"""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append(f"MODEL: {model_name.upper()}")
    lines.append("=" * 80)

    # Legitimate apps (label 0)
    legit_data = distribution['legitimate']
    lines.append(f"\nLEGITIMATE APPS (actual label = 0)")
    lines.append(f"Total samples: {legit_data['total_samples']:,}")
    lines.append("-" * 80)

    # Find max percentage for scaling
    max_pct = max(bin_data['percentage'] for bin_data in legit_data['bins'].values())

    for bin_label in sorted(legit_data['bins'].keys()):
        bin_data = legit_data['bins'][bin_label]
        pct = bin_data['percentage']
        count = bin_data['count']

        # Scale bar relative to max
        if max_pct > 0:
            bar = generate_ascii_bar((pct / max_pct) * 100, max_width=40)
        else:
            bar = '░' * 40

        lines.append(f"{bin_label}:  {bar}  {pct:5.1f}% ({count:6,} apps)")

    # Malicious apps (label 1)
    mal_data = distribution['malicious']
    lines.append(f"\nMALICIOUS APPS (actual label = 1)")
    lines.append(f"Total samples: {mal_data['total_samples']:,}")
    lines.append("-" * 80)

    # Find max percentage for scaling
    max_pct = max(bin_data['percentage'] for bin_data in mal_data['bins'].values())

    for bin_label in sorted(mal_data['bins'].keys()):
        bin_data = mal_data['bins'][bin_label]
        pct = bin_data['percentage']
        count = bin_data['count']

        # Scale bar relative to max
        if max_pct > 0:
            bar = generate_ascii_bar((pct / max_pct) * 100, max_width=40)
        else:
            bar = '░' * 40

        lines.append(f"{bin_label}:  {bar}  {pct:5.1f}% ({count:6,} apps)")

    return "\n".join(lines)


def generate_comparison_table(all_distributions):
    """Generate comparison table showing key bins across all models"""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("COMPARISON: HIGH-CONFIDENCE PREDICTIONS ACROSS ALL MODELS")
    lines.append("=" * 80)
    lines.append("\nPercentage of apps with confidence score 0.9-1.0 (very confident predictions)")
    lines.append("-" * 80)

    # Table header
    lines.append(f"{'Model':<15} {'Legitimate (0.9-1.0)':<25} {'Malicious (0.9-1.0)':<25}")
    lines.append("-" * 80)

    # Sort models by name
    for model_name in sorted(all_distributions.keys()):
        dist = all_distributions[model_name]

        legit_high = dist['legitimate']['bins']['0.9-1.0']['percentage']
        mal_high = dist['malicious']['bins']['0.9-1.0']['percentage']

        lines.append(f"{model_name:<15} {legit_high:6.2f}%                  {mal_high:6.2f}%")

    # Low confidence comparison
    lines.append("\n" + "=" * 80)
    lines.append("COMPARISON: LOW-CONFIDENCE PREDICTIONS (0.0-0.1)")
    lines.append("=" * 80)
    lines.append("\nPercentage of apps with confidence score 0.0-0.1 (very uncertain predictions)")
    lines.append("-" * 80)

    lines.append(f"{'Model':<15} {'Legitimate (0.0-0.1)':<25} {'Malicious (0.0-0.1)':<25}")
    lines.append("-" * 80)

    for model_name in sorted(all_distributions.keys()):
        dist = all_distributions[model_name]

        legit_low = dist['legitimate']['bins']['0.0-0.1']['percentage']
        mal_low = dist['malicious']['bins']['0.0-0.1']['percentage']

        lines.append(f"{model_name:<15} {legit_low:6.2f}%                  {mal_low:6.2f}%")

    return "\n".join(lines)


def generate_decision_quality_analysis(all_distributions):
    """Analyze decision-making quality based on confidence distributions"""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("DECISION QUALITY ANALYSIS")
    lines.append("=" * 80)
    lines.append("\nIdeal model behavior:")
    lines.append("  - Legitimate apps: High % in 0.0-0.1 bin (low malware score)")
    lines.append("  - Malicious apps: High % in 0.9-1.0 bin (high malware score)")
    lines.append("-" * 80)

    # Calculate quality score
    quality_scores = {}
    for model_name, dist in all_distributions.items():
        # Good: legit apps scored low (0.0-0.1)
        legit_correct_conf = dist['legitimate']['bins']['0.0-0.1']['percentage']

        # Good: malicious apps scored high (0.9-1.0)
        mal_correct_conf = dist['malicious']['bins']['0.9-1.0']['percentage']

        # Average of both
        quality_score = (legit_correct_conf + mal_correct_conf) / 2
        quality_scores[model_name] = {
            'score': quality_score,
            'legit_low': legit_correct_conf,
            'mal_high': mal_correct_conf
        }

    # Sort by quality score
    sorted_quality = sorted(quality_scores.items(), key=lambda x: x[1]['score'], reverse=True)

    lines.append(f"\n{'Rank':<6} {'Model':<15} {'Quality Score':<15} {'Legit Low %':<15} {'Mal High %':<15}")
    lines.append("-" * 80)

    for rank, (model_name, metrics) in enumerate(sorted_quality, 1):
        quality = metrics['score']
        legit_low = metrics['legit_low']
        mal_high = metrics['mal_high']

        emoji = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        lines.append(f"{emoji} {rank:<3} {model_name:<15} {quality:6.2f}%         {legit_low:6.2f}%         {mal_high:6.2f}%")

    return "\n".join(lines)


def generate_markdown_report(all_distributions, output_file='confidence_distributions_report.md'):
    """Generate markdown report"""
    with open(output_file, 'w') as f:
        f.write("# Confidence Score Distribution Analysis - All Models\n\n")
        f.write("This report shows the distribution of prediction confidence scores for all 7 models.\n\n")

        # Summary table
        f.write("## Summary: High-Confidence Predictions\n\n")
        f.write("| Model | Legitimate (0.9-1.0) | Malicious (0.9-1.0) |\n")
        f.write("|-------|---------------------:|--------------------:|\n")

        for model_name in sorted(all_distributions.keys()):
            dist = all_distributions[model_name]
            legit_high = dist['legitimate']['bins']['0.9-1.0']['percentage']
            mal_high = dist['malicious']['bins']['0.9-1.0']['percentage']
            f.write(f"| {model_name} | {legit_high:.2f}% | {mal_high:.2f}% |\n")

        # Full distributions
        for model_name in sorted(all_distributions.keys()):
            dist = all_distributions[model_name]
            f.write(f"\n## {model_name.upper()}\n\n")

            # Legitimate
            f.write(f"### Legitimate Apps (Label 0)\n\n")
            f.write("| Confidence Bin | Count | Percentage |\n")
            f.write("|---------------:|------:|-----------:|\n")

            for bin_label in sorted(dist['legitimate']['bins'].keys()):
                bin_data = dist['legitimate']['bins'][bin_label]
                f.write(f"| {bin_label} | {bin_data['count']:,} | {bin_data['percentage']:.2f}% |\n")

            # Malicious
            f.write(f"\n### Malicious Apps (Label 1)\n\n")
            f.write("| Confidence Bin | Count | Percentage |\n")
            f.write("|---------------:|------:|-----------:|\n")

            for bin_label in sorted(dist['malicious']['bins'].keys()):
                bin_data = dist['malicious']['bins'][bin_label]
                f.write(f"| {bin_label} | {bin_data['count']:,} | {bin_data['percentage']:.2f}% |\n")

    return output_file


def main():
    # Load data
    input_file = 'confidence_distributions_all_models.json'

    try:
        with open(input_file, 'r') as f:
            all_distributions = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_file} not found!")
        print("Please run extract_confidence_docker.py first")
        sys.exit(1)

    print("=" * 80)
    print("CONFIDENCE SCORE DISTRIBUTION VISUALIZATION")
    print("=" * 80)
    print(f"\nLoaded distributions for {len(all_distributions)} models")

    # Generate visualizations for each model
    for model_name in sorted(all_distributions.keys()):
        visualization = visualize_model_distribution(model_name, all_distributions[model_name])
        print(visualization)

    # Generate comparison table
    comparison = generate_comparison_table(all_distributions)
    print(comparison)

    # Generate decision quality analysis
    quality_analysis = generate_decision_quality_analysis(all_distributions)
    print(quality_analysis)

    # Generate markdown report
    md_file = generate_markdown_report(all_distributions)
    print(f"\n\nMarkdown report saved to: {md_file}")

    # Save text report
    text_report_file = 'confidence_distributions_report.txt'
    with open(text_report_file, 'w') as f:
        f.write("CONFIDENCE SCORE DISTRIBUTION VISUALIZATION\n")
        f.write("=" * 80 + "\n\n")

        for model_name in sorted(all_distributions.keys()):
            visualization = visualize_model_distribution(model_name, all_distributions[model_name])
            f.write(visualization + "\n")

        f.write(comparison + "\n")
        f.write(quality_analysis + "\n")

    print(f"Text report saved to: {text_report_file}")

    print("\n" + "=" * 80)
    print("COMPLETED")
    print("=" * 80)


if __name__ == '__main__':
    main()
