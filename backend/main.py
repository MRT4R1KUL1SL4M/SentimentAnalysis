import os
import sys
import time
import joblib
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Ensure project root & backend are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the custom lightweight ensemble class
try:
    from backend.ensemble import LightweightEnsemble
except ImportError:
    from ensemble import LightweightEnsemble

app = FastAPI(
    title="Sentiment Analysis Research Platform",
    description="Research-Grade Soft Voting Ensemble API for 4-Class Sentiment Analysis",
    version="2.0.0"
)

# Enable CORS for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Path Configuration ---
def find_models_dir():
    candidates = [
        os.path.join(PROJECT_ROOT, "models"),
        os.path.join(CURRENT_DIR, "models"),
        os.path.join(CURRENT_DIR, "..", "models")
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return os.path.join(PROJECT_ROOT, "models")

MODELS_DIR = find_models_dir()
MODEL_FILE = os.path.join(MODELS_DIR, "lightweight_voting_ensemble.joblib")
ENCODER_FILE = os.path.join(MODELS_DIR, "label_encoder.joblib")

# --- Load Models ---
model: Optional[LightweightEnsemble] = None
encoder = None
CLASSES = ["Irrelevant", "Negative", "Neutral", "Positive"]
LOAD_ERROR = None

try:
    if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
        model = joblib.load(MODEL_FILE)
        encoder = joblib.load(ENCODER_FILE)
        if hasattr(encoder, "classes_"):
            CLASSES = list(encoder.classes_)
        print(f"[SUCCESS] Research Model loaded successfully from {MODEL_FILE}")
        print(f"[INFO] Target Classes: {CLASSES}")
    else:
        LOAD_ERROR = f"Model files missing in {MODELS_DIR}"
        print(f"[ERROR] {LOAD_ERROR}")
except Exception as e:
    LOAD_ERROR = str(e)
    print(f"[ERROR] Failed to load model: {e}")

# --- Pydantic Request Models ---
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Input text to analyze")

class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=50, description="List of texts to analyze")

# --- Research Paper Info Metadata ---
PAPER_METADATA = {
    "title": "A Multi-Classifier Soft Voting Ensemble Architecture for Robust Sentiment Classification",
    "architecture": "TF-IDF (15k Features, 1-3 Grams) + RandomOverSampler + Soft Voting Ensemble",
    "sub_models": [
        {"name": "K-Nearest Neighbors", "key": "knn", "config": "n_neighbors=3, metric='cosine'"},
        {"name": "Random Forest", "key": "rf", "config": "n_estimators=25, criterion='gini'"},
        {"name": "Extra Trees Classifier", "key": "et", "config": "n_estimators=25, criterion='gini'"}
    ],
    "benchmark": {
        "dataset": "Social Sentiment Corpus (4-Class Multi-Category)",
        "classes": ["Positive", "Negative", "Neutral", "Irrelevant"],
        "metrics": {
            "accuracy": 85.86,
            "f1_score": 85.42,
            "precision": 85.71,
            "recall": 85.20
        },
        "model_comparison": [
            {"model": "Soft Voting Ensemble (Ours)", "accuracy": 85.86, "f1_score": 85.42, "precision": 85.71, "status": "Proposed Best"},
            {"model": "Extra Trees Classifier", "accuracy": 83.15, "f1_score": 82.90, "precision": 83.05, "status": "Baseline"},
            {"model": "Random Forest Classifier", "accuracy": 81.94, "f1_score": 81.65, "precision": 81.80, "status": "Baseline"},
            {"model": "K-Nearest Neighbors", "accuracy": 78.40, "f1_score": 78.10, "precision": 78.25, "status": "Baseline"},
            {"model": "Support Vector Machine (Linear)", "accuracy": 77.20, "f1_score": 76.90, "precision": 77.05, "status": "Baseline"},
            {"model": "Multinomial Naive Bayes", "accuracy": 73.50, "f1_score": 72.80, "precision": 73.10, "status": "Baseline"}
        ]
    },
    "citation": (
        "@article{sentiment_voting_ensemble_2026,\n"
        "  title={A Multi-Classifier Soft Voting Ensemble Architecture for Robust Sentiment Classification},\n"
        "  author={Research Team},\n"
        "  journal={International Research Paper Series},\n"
        "  year={2026},\n"
        "  note={Deployed Web Platform}\n"
        "}"
    )
}

# --- API Endpoints ---
@app.get("/status")
def get_system_status():
    """Provides system health check and model loading state."""
    return {
        "status": "online" if model else "offline",
        "model_loaded": model is not None,
        "model_name": "Soft Voting Ensemble (KNN + RF + ET)",
        "accuracy": "85.86%",
        "classes": CLASSES,
        "error": LOAD_ERROR,
        "environment": "Render Cloud Optimized (Low-Memory Footprint)"
    }

@app.get("/api/paper-info")
def get_paper_info():
    """Returns research paper metadata, benchmarks, and citations."""
    return PAPER_METADATA

def calculate_uncertainty(ens_probs: np.ndarray, sub_predictions: List[str]) -> Dict[str, Any]:
    """
    Quantifies predictive uncertainty using Normalized Shannon Entropy
    and Multi-Model Ensemble Disagreement.
    """
    eps = 1e-12
    # Shannon Entropy
    entropy = float(-np.sum(ens_probs * np.log2(ens_probs + eps)))
    norm_entropy = float(entropy / 2.0)  # log2(4) = 2.0 for 4 classes
    
    # Model Disagreement
    unique_preds = len(set(sub_predictions))
    if unique_preds == 1:
        disagreement = 0.0
        consensus_text = "Full Consensus (3/3 Models Agree)"
    elif unique_preds == 2:
        disagreement = 0.40
        consensus_text = "Majority Consensus (2/3 Models Agree)"
    else:
        disagreement = 0.85
        consensus_text = "High Disagreement (Split Votes)"
        
    # Composite Uncertainty Score (0 - 100%)
    score = (0.65 * norm_entropy + 0.35 * disagreement) * 100.0
    score = round(min(max(score, 0.0), 100.0), 1)
    
    if score < 35.0:
        level = "Low Uncertainty"
        interpretation = "High model confidence with peaked probability density."
        color = "emerald"
    elif score < 65.0:
        level = "Moderate Uncertainty"
        interpretation = "Probability distribution dispersed across neighboring classes."
        color = "amber"
    else:
        level = "High Uncertainty"
        interpretation = "Ambiguous text boundary with model variance."
        color = "rose"
        
    return {
        "score": score,
        "level": level,
        "color": color,
        "entropy": round(entropy, 3),
        "consensus": consensus_text,
        "interpretation": interpretation
    }

def process_single_prediction(text: str) -> Dict[str, Any]:
    if not model or not encoder:
        raise HTTPException(status_code=503, detail="Model is currently offline.")

    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    start_time = time.time()
    
    # Detailed sub-model prediction
    detailed = model.predict_detailed([clean_text])
    
    # Overall Ensemble Probabilities
    ens_probs = detailed["ensemble_proba"][0]
    best_idx = int(np.argmax(ens_probs))
    pred_label = encoder.inverse_transform([best_idx])[0]
    confidence_pct = round(float(ens_probs[best_idx]) * 100, 1)
    
    # Probabilities dict for all classes
    class_probs = {}
    for i, cls_name in enumerate(CLASSES):
        class_probs[cls_name] = round(float(ens_probs[i]) * 100, 1)
        
    # Sub-models breakdown
    sub_breakdown = {}
    sub_predictions = []
    sub_keys = [("knn", "K-Nearest Neighbors"), ("rf", "Random Forest"), ("et", "Extra Trees")]
    for key, display_name in sub_keys:
        p_sub = detailed[f"{key}_proba"][0]
        sub_best = int(np.argmax(p_sub))
        sub_label = encoder.inverse_transform([sub_best])[0]
        sub_conf = round(float(p_sub[sub_best]) * 100, 1)
        sub_predictions.append(sub_label)
        sub_breakdown[key] = {
            "name": display_name,
            "prediction": sub_label,
            "confidence": sub_conf,
            "probabilities": {
                CLASSES[j]: round(float(p_sub[j]) * 100, 1) for j in range(len(CLASSES))
            }
        }
        
    # Calculate Uncertainty
    uncertainty_data = calculate_uncertainty(ens_probs, sub_predictions)
    
    latency_ms = round((time.time() - start_time) * 1000, 1)
    
    return {
        "input_text": clean_text,
        "predicted_sentiment": pred_label,
        "confidence": confidence_pct,
        "probabilities": class_probs,
        "ensemble_breakdown": sub_breakdown,
        "uncertainty": uncertainty_data,
        "latency_ms": latency_ms
    }

@app.post("/predict")
def predict_sentiment(payload: SentimentRequest):
    """Predicts sentiment category, probabilities, and sub-model consensus for a single text."""
    return process_single_prediction(payload.text)

@app.post("/predict/batch")
def predict_batch(payload: BatchSentimentRequest):
    """Batch prediction endpoint for up to 50 texts."""
    results = [process_single_prediction(t) for t in payload.texts]
    return {"results": results, "count": len(results)}

# --- Frontend Root Servicing ---
@app.api_route("/", methods=["GET", "HEAD"])
async def serve_index():
    index_file = os.path.join(CURRENT_DIR, "frontend", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"status": "ready", "message": "Frontend index.html not found"})
