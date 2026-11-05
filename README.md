# SentiQuant

> **Sentiment Analysis & Predictive Uncertainty Quantification Engine**

<p align="left">
  <a href="https://sentimentanalysis-81an.onrender.com/" target="_blank">
    <img src="https://img.shields.io/badge/Live%20Demo-Open%20Web%20App%20%E2%86%92-10b981?style=for-the-badge&logo=render&logoColor=white" alt="Live Demo" height="34" />
  </a>
</p>

🔗 **Live URL:** [https://sentimentanalysis-81an.onrender.com/](https://sentimentanalysis-81an.onrender.com/)

---

## Overview

**SentiQuant** is an intelligent Natural Language Processing (NLP) web platform that combines multi-model sentiment classification with **Predictive Uncertainty Quantification**.

Instead of treating classification as a black box, SentiQuant evaluates text across four distinct classes (`Positive`, `Negative`, `Neutral`, and `Irrelevant`), providing both a final consensus verdict and an empirical measure of model confidence.

---

## How It Works

1. **Feature Extraction**: Input text is processed through a sublinear TF-IDF vectorizer (15,000 features, 1–3 n-grams).
2. **Soft Voting Tri-Ensemble**: Combines the probabilistic outputs of three parallel estimators:
   - **K-Nearest Neighbors (KNN)**
   - **Random Forest**
   - **Extra Trees**
3. **Predictive Uncertainty Quantification**:
   - Calculates **Normalized Shannon Entropy** from the probability distribution.
   - Evaluates **Inter-Model Agreement** across the three sub-models.
   - Computes a composite **Uncertainty Dispersion Score** (Low, Moderate, High) to gauge prediction reliability.

---

## Core Features

- **4-Class Sentiment Analysis**: Categorizes text into `Positive`, `Negative`, `Neutral`, or `Irrelevant`.
- **Predictive Uncertainty**: Displays dispersion and entropy metrics to highlight ambiguous or edge-case text.
- **Sub-Model Inspection**: View individual predictions and confidence scores for KNN, Random Forest, and Extra Trees.
- **Cloud Optimized**: Compact model footprint (~49.8 MB) designed to run smoothly within a 512 MB RAM footprint.
- **Modern Minimalist UI**: Sleek, distraction-free scientific interface with real-time inference.

---
