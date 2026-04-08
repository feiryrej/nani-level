"""
app.py - FastAPI backend for Japanese Sentence Difficulty Classifier
COSC 402 - AI/ML Implementation

Run with: uvicorn app:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize the FastAPI app with metadata for auto-generated API docs (/docs)
app = FastAPI(
    title="Nihongo Difficulty Classifier",
    description="Classifies Japanese text as Easy, Intermediate, or Hard using ML",
    version="1.0.0"
)

# Allow the frontend (HTML file opened in browser) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
# Pydantic models enforce type validation and auto-generate API schema docs.

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict
    matched_vocab: list
    char_count: int

# Routes

@app.get("/")
def root():
    return {"message": "Nihongo Difficulty Classifier API is running"}

@app.post("/predict", response_model=PredictResponse)
def predict_difficulty(request: PredictRequest):
    """
    Classify the difficulty of a Japanese sentence.
    Returns: label (easy/intermediate/hard), confidence %, probabilities, and matched vocab.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if len(text) > 500:
        raise HTTPException(status_code=400, detail="Text too long (max 500 characters).")

    try:
        from core.classifier import predict
        result = predict(text)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok"}