# Android Package Manager Info Filtering

A machine learning system for classifying Android package names as legitimate or suspicious/malicious. Designed to filter suspicious apps early in the analysis pipeline, reducing computational load on heavy analysis operations. This is only a single or atomic part of a larger Machine Learning Architecture Proof of Concept for Filtering Android Application, the basic idea is to utilize the Android's Built-In Package Manager Information Readily available for decision making for identifying hignly suspicious Application for further deep analysis, requiring heavy operation such as unpacking the applications, parsing operations of dex and other compenents. Refer to the [Package Manager API Reference](PackageManager_API_Reference.md) for list of available information that can be obtained.

---
---
# Documents
- **[MODEL_EVALUATION](MODEL_EVALUATION.md)**
  - Comprehensive Model Evaluation and Comparison
- **[Feature-EnhancedLSTM_vs_CNN_ConfidenceScore.md](Feature-EnhancedLSTM_vs_CNN_ConfidenceScore.md)** 
  - Threshold analysis & confidence scores between best 2 Candidates.
- **[evaluation/COMPREHENSIVE_THRESHOLD_ANALYSIS.md](evaluation/COMPREHENSIVE_THRESHOLD_ANALYSIS.md)** 
  - Threshold Analysis on Each Models

---
# Notes:

- Modernize for TF2.0 and rerun To Add Transformer in Comparison.
- Transformer is a relatively new (2024) model, while transformer is designed and best applied for Large Data Sets, it has shown good performance, obviously the model is not the best appleid in this case, added for the sake seeing of how transformer performs.
- xgboost is added as baseline for stacking xgboost with lstm, these stacked models are not included here and requires GPU for training.
- base_model is only an artifacts, it uses a different vocabulary, its practically the older version of the dummy model but, this new dummy model is synchronized with other model to use the same vocabulary.
