import numpy as np

class LightweightEnsemble:
    """
    Research-Grade Soft Voting Ensemble Classifier
    Combines:
    1. K-Nearest Neighbors (metric='cosine', n_neighbors=3)
    2. Random Forest Classifier (Pruned n_estimators=25)
    3. Extra Trees Classifier (Pruned n_estimators=25)
    Using shared TF-IDF feature extraction (15,000 max features, n-grams 1-3).
    """
    def __init__(self, tfidf, knn, rf, et, weights=None):
        self.tfidf = tfidf
        self.knn = knn
        self.rf = rf
        self.et = et
        self.weights = weights if weights is not None else [1.0, 1.0, 1.0]
        self.classes_ = None

    def transform_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.tfidf.transform(texts)

    def predict_proba(self, texts):
        X = self.transform_text(texts)
        p_knn = self.knn.predict_proba(X)
        p_rf = self.rf.predict_proba(X)
        p_et = self.et.predict_proba(X)
        w = np.array(self.weights, dtype=float) / sum(self.weights)
        return w[0] * p_knn + w[1] * p_rf + w[2] * p_et

    def predict_detailed(self, texts):
        X = self.transform_text(texts)
        p_knn = self.knn.predict_proba(X)
        p_rf = self.rf.predict_proba(X)
        p_et = self.et.predict_proba(X)
        w = np.array(self.weights, dtype=float) / sum(self.weights)
        p_ensemble = w[0] * p_knn + w[1] * p_rf + w[2] * p_et

        return {
            "ensemble_proba": p_ensemble,
            "knn_proba": p_knn,
            "rf_proba": p_rf,
            "et_proba": p_et
        }

    def predict(self, texts):
        probs = self.predict_proba(texts)
        return np.argmax(probs, axis=1)

