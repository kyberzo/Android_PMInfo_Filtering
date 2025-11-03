#!/usr/bin/env python
"""
Feature Engineering Utilities for Android Package Name Classification

Extracts domain-specific features from package names to improve model accuracy.

Features:
1. Entropy-based: Shannon entropy, bigram analysis
2. Structural: Segment analysis, pattern detection
3. Character: Case mixing, digit patterns
4. Linguistic: Dictionary matching

"""

import numpy as np
from collections import Counter
import math


# ============================================================================
# CATEGORY 1: ENTROPY & RANDOMNESS FEATURES
# ============================================================================

def shannon_entropy(text):
    """
    Calculate Shannon entropy of text.

    Higher entropy = more random = more suspicious

    Range: 0 (all same char) to ~5.75 (perfectly random)

    Examples:
    - "com.google.android.maps" → ~4.2 (legitimate - has structure)
    - "AKDIJG.DFFYDWER" → ~4.8 (suspicious - random)
    """
    if not text:
        return 0.0

    counts = Counter(text)
    total = len(text)
    probabilities = [count / total for count in counts.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)

    return float(entropy)


def get_bigram_features(package_name):
    """
    Extract bigram (2-character sequence) patterns.

    Legitimate packages have common bigrams (th, er, io, com)
    Suspicious packages have random bigrams

    Returns dict with:
    - bigram_diversity: unique bigrams / total bigrams (high = suspicious)
    - bigram_repetition: max repeated bigram / total (high = legitimate)
    """
    if len(package_name) < 2:
        return {
            'bigram_diversity': 0.0,
            'bigram_repetition': 0.0
        }

    bigrams = [package_name[i:i+2] for i in range(len(package_name)-1)]
    bigram_counts = Counter(bigrams)

    # Diversity: how many unique bigrams vs total
    unique_bigrams = len(set(bigrams))
    bigram_diversity = unique_bigrams / len(bigrams) if bigrams else 0.0

    # Repetition: highest repeated bigram frequency
    max_bigram_freq = max(bigram_counts.values()) if bigram_counts else 0
    repetition_score = max_bigram_freq / len(bigrams) if bigrams else 0.0

    return {
        'bigram_diversity': float(bigram_diversity),
        'bigram_repetition': float(repetition_score)
    }


# ============================================================================
# CATEGORY 2: STRUCTURAL FEATURES
# ============================================================================

def get_segment_features(package_name):
    """
    Analyze package structure by segments (divided by dots).

    Legitimate: well-formed segments with consistent lengths
    Suspicious: irregular, random segments

    Returns dict with:
    - num_segments: number of dot-separated parts
    - avg_segment_length: average characters per segment
    - segment_length_variance: variance in segment lengths
    - avg_segment_entropy: average entropy per segment
    """
    segments = package_name.split('.')

    if not segments:
        return {
            'num_segments': 0,
            'avg_segment_length': 0.0,
            'segment_length_variance': 0.0,
            'avg_segment_entropy': 0.0
        }

    # Segment length statistics
    segment_lengths = [len(seg) for seg in segments]
    avg_segment_length = np.mean(segment_lengths) if segment_lengths else 0.0
    segment_length_variance = np.var(segment_lengths) if segment_lengths else 0.0

    # Segment randomness
    segment_entropies = [shannon_entropy(seg) for seg in segments]
    avg_entropy = np.mean(segment_entropies) if segment_entropies else 0.0

    return {
        'num_segments': float(len(segments)),
        'avg_segment_length': float(avg_segment_length),
        'segment_length_variance': float(segment_length_variance),
        'avg_segment_entropy': float(avg_entropy)
    }


def get_pattern_features(package_name):
    """
    Detect known legitimate and suspicious patterns.

    Legitimate packages use real company names and Android keywords
    Suspicious packages use common obfuscation patterns

    Returns dict with boolean flags (0 or 1)
    """
    # Known legitimate patterns
    LEGITIMATE_COMPANIES = {
        'com', 'org', 'android', 'androidx', 'google', 'facebook', 'amazon',
        'twitter', 'samsung', 'microsoft', 'apple', 'adobe', 'spotify'
    }

    LEGITIMATE_KEYWORDS = {
        'app', 'apps', 'service', 'widget', 'receiver', 'provider', 'activity',
        'manager', 'handler', 'client', 'server', 'database', 'config',
        'util', 'helper', 'cache', 'sync', 'notification', 'settings',
        'launcher', 'browser', 'keyboard', 'gallery', 'music', 'video',
        'mail', 'chat', 'call', 'camera', 'photo', 'file'
    }

    # Known suspicious patterns
    SUSPICIOUS_INDICATORS = [
        '.view', '.activity', '.service', '.intent',
        'sdk', 'ads', 'analytics', 'tracking', 'click'
    ]

    segments = package_name.lower().split('.')

    # Check for legitimate patterns
    has_legit_company = any(seg in LEGITIMATE_COMPANIES for seg in segments)
    has_legit_keyword = any(seg in LEGITIMATE_KEYWORDS for seg in segments)

    # Check for suspicious patterns
    has_suspicious = any(susp in package_name.lower() for susp in SUSPICIOUS_INDICATORS)

    return {
        'has_legitimate_company': float(has_legit_company),
        'has_legitimate_keyword': float(has_legit_keyword),
        'has_suspicious_pattern': float(has_suspicious)
    }


# ============================================================================
# CATEGORY 3: CHARACTER COMPOSITION FEATURES
# ============================================================================

def get_case_features(package_name):
    """
    Analyze character case patterns and mixing.

    Legitimate: consistent casing (usually all lowercase)
    Suspicious: random mixing of upper/lower case

    Returns dict with case-related statistics
    """
    if not package_name:
        return {
            'uppercase_ratio': 0.0,
            'lowercase_ratio': 0.0,
            'digit_ratio': 0.0,
            'special_ratio': 0.0,
            'case_alternations': 0.0,
            'is_all_upper': 0.0,
            'is_all_lower': 0.0
        }

    uppercase_count = sum(1 for c in package_name if c.isupper())
    lowercase_count = sum(1 for c in package_name if c.islower())
    digit_count = sum(1 for c in package_name if c.isdigit())
    special_count = sum(1 for c in package_name if not c.isalnum())

    total_chars = len(package_name)

    # Case patterns
    is_all_upper = uppercase_count > 0 and lowercase_count == 0
    is_all_lower = lowercase_count > 0 and uppercase_count == 0

    # Case alternations: count transitions between upper and lower
    alternations = 0
    for i in range(len(package_name)-1):
        curr = package_name[i]
        next_char = package_name[i+1]
        if curr.isalpha() and next_char.isalpha():
            if curr.isupper() != next_char.isupper():
                alternations += 1

    return {
        'uppercase_ratio': float(uppercase_count / total_chars),
        'lowercase_ratio': float(lowercase_count / total_chars),
        'digit_ratio': float(digit_count / total_chars),
        'special_ratio': float(special_count / total_chars),
        'case_alternations': float(alternations),
        'is_all_upper': float(is_all_upper),
        'is_all_lower': float(is_all_lower)
    }


def get_char_patterns(package_name):
    """
    Analyze digit and special character patterns.

    Legitimate: digits grouped or absent
    Suspicious: digits scattered throughout

    Returns dict with digit clustering and distribution
    """
    segments = package_name.split('.')

    if not segments:
        return {
            'digit_segment_ratio': 0.0,
            'digit_clustering': 0.0,
            'has_consecutive_digits': 0.0
        }

    segments_with_digits = sum(1 for seg in segments if any(c.isdigit() for c in seg))

    # Digit clustering: measure spacing between digits
    digit_positions = [i for i, c in enumerate(package_name) if c.isdigit()]
    digit_clustering = 0.0
    if len(digit_positions) > 1:
        gaps = [digit_positions[i+1] - digit_positions[i]
                for i in range(len(digit_positions)-1)]
        digit_clustering = float(max(gaps)) if gaps else 0.0

    # Check for consecutive digits
    has_consecutive = any(package_name[i:i+2].isdigit()
                         for i in range(len(package_name)-1))

    return {
        'digit_segment_ratio': float(segments_with_digits / len(segments)) if segments else 0.0,
        'digit_clustering': digit_clustering,
        'has_consecutive_digits': float(has_consecutive)
    }


# ============================================================================
# CATEGORY 4: LINGUISTIC FEATURES
# ============================================================================

def get_dictionary_features(package_name):
    """
    Check if package segments match real English words.

    Legitimate: high ratio of real words
    Suspicious: mostly gibberish

    Returns dict with word matching statistics
    """
    # Common words found in legitimate package names
    COMMON_WORDS = {
        'android', 'app', 'apps', 'service', 'activity', 'google', 'facebook',
        'amazon', 'twitter', 'system', 'widget', 'receiver', 'provider',
        'manager', 'handler', 'client', 'server', 'database', 'config',
        'util', 'helper', 'cache', 'sync', 'notification', 'settings',
        'launcher', 'browser', 'keyboard', 'gallery', 'music', 'video',
        'mail', 'chat', 'call', 'camera', 'photo', 'file', 'free',
        'pro', 'lite', 'beta', 'demo', 'test', 'dev', 'debug'
    }

    segments = package_name.lower().split('.')

    if not segments:
        return {'real_word_ratio': 0.0}

    # Count segments that are real words or digits (often acceptable)
    real_word_segments = sum(1 for seg in segments
                            if seg in COMMON_WORDS or seg.isdigit())

    return {
        'real_word_ratio': float(real_word_segments / len(segments))
    }


# ============================================================================
# COMPOSITE FEATURE EXTRACTION
# ============================================================================

def extract_all_features(package_name):
    """
    Extract ALL engineered features from package name.

    Returns: dict with ~30 features (most in range 0-1)

    Total feature count:
    - Entropy: 3
    - Segment: 4
    - Pattern: 3
    - Case: 7
    - Char: 3
    - Dictionary: 1
    = 21 features total
    """
    features = {}

    # Category 1: Entropy features
    entropy_features = {
        'shannon_entropy': shannon_entropy(package_name),
    }
    entropy_features.update(get_bigram_features(package_name))
    features.update(entropy_features)

    # Category 2: Structural features
    features.update(get_segment_features(package_name))
    features.update(get_pattern_features(package_name))

    # Category 3: Character composition
    features.update(get_case_features(package_name))
    features.update(get_char_patterns(package_name))

    # Category 4: Linguistic features
    features.update(get_dictionary_features(package_name))

    return features


def extract_features_to_array(package_name, feature_names=None):
    """
    Extract all features and return as numpy array.

    Args:
        package_name: Android package name string
        feature_names: If provided, only extract these features (for consistency)

    Returns:
        np.array of feature values (will be float32)
    """
    features = extract_all_features(package_name)

    if feature_names is None:
        # Use sorted keys for consistent ordering
        feature_names = sorted(features.keys())

    # Extract values in consistent order
    feature_values = [features.get(fname, 0.0) for fname in feature_names]

    return np.array(feature_values, dtype=np.float32), feature_names


def normalize_features(feature_array):
    """
    Normalize features to 0-1 range using min-max scaling.

    Note: For production, compute min/max on training set first!

    Args:
        feature_array: np.array of shape (batch_size, num_features)

    Returns:
        Normalized array (same shape)
    """
    # Simple min-max normalization
    min_vals = np.min(feature_array, axis=0)
    max_vals = np.max(feature_array, axis=0)

    # Avoid division by zero
    ranges = np.where(max_vals - min_vals > 0, max_vals - min_vals, 1.0)

    normalized = (feature_array - min_vals) / ranges

    return normalized, min_vals, ranges


# ============================================================================
# FEATURE STATISTICS
# ============================================================================

def get_feature_stats(package_names_list, labels=None):
    """
    Calculate statistics for all features across multiple package names.

    Useful for understanding feature distributions and importance.

    Args:
        package_names_list: List of package name strings
        labels: Optional list of labels (0 or 1) for per-class analysis

    Returns:
        dict with statistics per feature
    """
    all_features = []

    for pkg in package_names_list:
        features = extract_all_features(pkg)
        all_features.append(features)

    # Convert to array for analysis
    feature_names = sorted(all_features[0].keys())
    feature_matrix = np.array([[f[name] for name in feature_names]
                              for f in all_features])

    stats = {}

    for i, fname in enumerate(feature_names):
        col = feature_matrix[:, i]
        stats[fname] = {
            'mean': float(np.mean(col)),
            'std': float(np.std(col)),
            'min': float(np.min(col)),
            'max': float(np.max(col)),
            'median': float(np.median(col))
        }

        # Per-class statistics if labels provided
        if labels is not None:
            legit_mask = np.array(labels) == 0
            susp_mask = np.array(labels) == 1

            if np.any(legit_mask):
                stats[fname]['legit_mean'] = float(np.mean(col[legit_mask]))
            if np.any(susp_mask):
                stats[fname]['susp_mean'] = float(np.mean(col[susp_mask]))

    return stats


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    # Test with examples
    legit_pkg = "com.google.android.apps.maps"
    susp_pkg = "AKDIJG.DFFYDWER"

    print("=" * 70)
    print("FEATURE EXTRACTION EXAMPLES")
    print("=" * 70)
    print()

    print(f"Legitimate: {legit_pkg}")
    legit_features = extract_all_features(legit_pkg)
    for name, value in sorted(legit_features.items()):
        print(f"  {name:.<40} {value:.4f}")

    print()
    print(f"Suspicious: {susp_pkg}")
    susp_features = extract_all_features(susp_pkg)
    for name, value in sorted(susp_features.items()):
        print(f"  {name:.<40} {value:.4f}")

    print()
    print("Key Differences:")
    print(f"  Shannon Entropy (legit < susp):    {legit_features['shannon_entropy']:.4f} < {susp_features['shannon_entropy']:.4f}")
    print(f"  Real Words (legit > susp):          {legit_features['real_word_ratio']:.4f} > {susp_features['real_word_ratio']:.4f}")
    print(f"  Case Alternations (legit < susp):  {legit_features['case_alternations']:.4f} < {susp_features['case_alternations']:.4f}")
    print()
