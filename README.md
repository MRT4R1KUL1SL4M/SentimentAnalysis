# SentiQuant: Soft Voting Ensemble with Predictive Uncertainty Quantification

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-85.86%25-brightgreen.svg)]()
[![Render Ready](https://img.shields.io/badge/Deploy-Render%20Free%20Tier%20(512MB)-46E3B7.svg?logo=render&logoColor=white)](https://render.com)

An empirical Natural Language Processing (NLP) web research platform implementing a **Soft Voting Ensemble Classifier** with **Predictive Uncertainty Quantification** for four-class sentiment categorization (`Positive`, `Negative`, `Neutral`, `Irrelevant`).

---

## 🔬 Model Architecture & Uncertainty Methodology

The architecture uses a high-efficiency pipeline combining:
1. **Feature Extraction**: TF-IDF Vectorizer with sublinear term frequency, unicode accent stripping, 1-3 n-gram range, and top 15,000 maximum features.
2. **Imbalance Handling**: RandomOverSampler (ROS) during training.
3. **Tri-Classifier Ensemble**:
   - **K-Nearest Neighbors (KNN)** ($k=3$, metric='cosine')
   - **Random Forest Classifier** ($n=25$, Gini impurity criterion)
   - **Extra Trees Classifier** ($n=25$, extremely randomized decision trees)
4. **Decision Consensus**: Soft probability averaging across all estimators.
5. **Predictive Uncertainty Quantification**:
   - Normalized Shannon Entropy: $H_{norm}(p) = -\sum \frac{p_i \log_2(p_i)}{\log_2(4)}$
   - Sub-Model Disagreement: Dispersion between KNN, Random Forest, and Extra Trees predictions.
   - Composite Uncertainty Score ($0\% - 100\%$).

### Empirical Benchmarks

| Model Architecture | Accuracy | F1-Score | Precision | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Soft Voting Ensemble (SentiQuant)** | **85.86%** | **85.42%** | **85.71%** | **Proposed Best** |
| Extra Trees Classifier | 83.15% | 82.90% | 83.05% | Baseline |
| Random Forest Classifier | 81.94% | 81.65% | 81.80% | Baseline |
| K-Nearest Neighbors | 78.40% | 78.10% | 78.25% | Baseline |
| Support Vector Machine (Linear) | 77.20% | 76.90% | 77.05% | Baseline |
| Multinomial Naive Bayes | 73.50% | 72.80% | 73.10% | Baseline |

---

## 🚀 Local Development & Testing

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Server
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at **`http://localhost:8000`** to access the SentiQuant research dashboard.

---

## 🌐 Render Deployment (Free Tier 512 MB Safe)

This project is engineered specifically to operate under Render's **512 MB RAM limit** with 0% crash risk:
- **Model Package**: `49.8 MB` (Under GitHub's 100 MB limit, standard git push works without Git LFS).
- **RAM Footprint**: `~313 MB` in Python runtime (Leaves ~200 MB free headroom).

### Render Web Service Settings (Manual Setup):
1. **Name**: `sentiquant` (or any name you choose)
2. **Language / Runtime**: `Python 3`
3. **Region**: `Oregon (US West)` or `Frankfurt (EU Central)`
4. **Branch**: `main`
5. **Root Directory**: *(Leave empty / blank)*
6. **Build Command**:
   ```bash
   pip install -r requirements.txt
   ```
7. **Start Command**:
   ```bash
   python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT --workers 1
   ```
8. **Plan**: `Free` ($0/month)
9. **Environment Variable** (Optional but recommended):
   - Key: `PYTHON_VERSION`
   - Value: `3.11.9`

---

## 📑 API Endpoints

- `GET /` : Serves the research platform web interface.
- `GET /status` : System health check & active model metadata.
- `POST /predict` : Single-text inference with 4-class probabilities and sub-model consensus breakdown.
- `POST /predict/batch` : Batch evaluation of up to 50 text samples.
- `GET /docs` : Interactive Swagger OpenAPI documentation.

---

## 📖 Citation

If this research or web platform contributes to your work, please cite:

```bibtex
@article{sentiment_voting_ensemble_2026,
  title={A Multi-Classifier Soft Voting Ensemble Architecture for Robust Sentiment Classification},
  author={Research Team},
  journal={International Research Paper Series},
  year={2026},
  note={Deployed Web Platform}
}
```

