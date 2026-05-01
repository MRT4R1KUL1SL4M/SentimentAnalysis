import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os
from fastapi.responses import FileResponse

app = FastAPI(title="Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
# Set your model's display name and file name here
MODEL_NAME = "Voting Classier"
MODEL_FILE = "voting_classifier_knn_rf_et.joblib"
VECTORIZER_FILE = "tfidf_vectorizer.joblib"
ENCODER_FILE = "label_encoder.joblib"

# --- Load Models ---
model = None
vectorizer = None
encoder = None

try:
    model_path = os.path.join("..", "models", MODEL_FILE)
    vectorizer_path = os.path.join("..", "models", VECTORIZER_FILE)
    encoder_path = os.path.join("..", "models", ENCODER_FILE)

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    encoder = joblib.load(encoder_path)
    
    print(f"✅ Model '{MODEL_NAME}' loaded successfully.")

except FileNotFoundError:
    print(f"❌ CRITICAL ERROR: Model file not found.")
    model, vectorizer, encoder = None, None, None
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
    model, vectorizer, encoder = None, None, None

# --- Pydantic Model for Input ---
class TextInput(BaseModel):
    text: str

# --- API Endpoint (for prediction) ---
@app.post("/predict")
def predict_sentiment(input: TextInput):
    if not all([model, vectorizer, encoder]):
        return {"error": "Model components are not loaded."}

    try:
        text_data = [input.text]
        vectorized_text = vectorizer.transform(text_data)
        prediction_index = model.predict(vectorized_text)
        prediction_label = encoder.inverse_transform(prediction_index)

        return {
            "input_text": input.text,
            "predicted_sentiment": prediction_label[0]
        }
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

# --- Status Endpoint (for frontend) ---
@app.get("/status")
def get_status():
    """Provides the frontend with the current model name and status."""
    return {
        "model_name": MODEL_NAME if model else "Model Not Loaded",
        "status": "online" if model else "offline"
    }

# --- Frontend Servicing (to show index.html) ---
@app.get("/")
async def read_index():
    return FileResponse(os.path.join("frontend", "index.html"))