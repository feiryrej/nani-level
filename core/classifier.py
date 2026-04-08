"""
classifier.py - Prediction logic for Japanese sentence difficulty
Loads trained models and provides predict() function used by the API.
"""

import os
import pickle
import numpy as np

# Paths
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "..", "models")

RF_MODEL_PATH    = os.path.join(MODEL_DIR, "difficulty_model.pkl")
NN_MODEL_PATH    = os.path.join(MODEL_DIR, "nn_model.keras")
LOOKUP_PATH      = os.path.join(MODEL_DIR, "word_to_level.pkl")

LABEL_NAMES = ["easy", "intermediate", "hard"]

# Load models once at import time
_rf_bundle     = None
_nn_model      = None
_word_to_level = None

def _load_models():
    global _rf_bundle, _nn_model, _word_to_level

    if not os.path.exists(RF_MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Please run `python training/train.py` first."
        )

    with open(RF_MODEL_PATH, "rb") as f:
        _rf_bundle = pickle.load(f)

    with open(LOOKUP_PATH, "rb") as f:
        _word_to_level = pickle.load(f)

    # Load Keras model lazily (optional — only if file exists)
    if os.path.exists(NN_MODEL_PATH):
        import tensorflow as tf
        _nn_model = tf.keras.models.load_model(NN_MODEL_PATH)

_load_models()

# Feature extraction (must match training/train.py)

def _is_kanji(c):
    return '\u4e00' <= c <= '\u9fff'

def _is_hiragana(c):
    return '\u3041' <= c <= '\u309f'

def _is_katakana(c):
    return '\u30a0' <= c <= '\u30ff'

def _extract_features(text):
    text = str(text).strip()
    if not text:
        return [0.0] * 7

    total   = len(text)
    kanji   = sum(1 for c in text if _is_kanji(c))
    hira    = sum(1 for c in text if _is_hiragana(c))
    kata    = sum(1 for c in text if _is_katakana(c))

    kanji_ratio = kanji / total
    hira_ratio  = hira  / total
    kata_ratio  = kata  / total

    matched_levels = []
    for word, lvl in _word_to_level.items():
        if word in text and len(word) > 1:
            matched_levels.append(lvl)

    if matched_levels:
        avg_level = float(np.mean(matched_levels))
        min_level = float(min(matched_levels))
        num_words = float(len(matched_levels))
    else:
        avg_level = 1.0
        min_level = 1.0
        num_words = 0.0

    return [kanji_ratio, hira_ratio, kata_ratio, avg_level, min_level, num_words, float(total)]

# Public API

def predict(text: str) -> dict:
    """
    Predict the difficulty of a Japanese text.
    Returns a dict with label, confidence, and per-class probabilities.
    """
    features = np.array([_extract_features(text)])

    rf    = _rf_bundle["model"]
    scaler = _rf_bundle["scaler"]
    X_scaled = scaler.transform(features)

    rf_proba   = rf.predict_proba(X_scaled)[0]   # [easy, intermediate, hard]
    rf_label   = LABEL_NAMES[int(np.argmax(rf_proba))]
    rf_conf    = float(np.max(rf_proba))

    # Optional: blend with neural network if available
    if _nn_model is not None:
        nn_proba  = _nn_model.predict(X_scaled, verbose=0)[0]
        # Weighted ensemble: 60% RF, 40% NN
        blend     = 0.6 * rf_proba + 0.4 * nn_proba
        label     = LABEL_NAMES[int(np.argmax(blend))]
        confidence = float(np.max(blend))
        probabilities = {
            "easy":         float(blend[0]),
            "intermediate": float(blend[1]),
            "hard":         float(blend[2]),
        }
    else:
        label      = rf_label
        confidence = rf_conf
        probabilities = {
            "easy":         float(rf_proba[0]),
            "intermediate": float(rf_proba[1]),
            "hard":         float(rf_proba[2]),
        }

    # Gather matched vocabulary for transparency
    matched_vocab = []
    for word, lvl in _word_to_level.items():
        if word in text and len(word) > 1:
            matched_vocab.append({"word": word, "jlpt_level": lvl})
    matched_vocab = sorted(matched_vocab, key=lambda x: x["jlpt_level"])[:10]

    return {
        "label":         label,
        "confidence":    round(confidence * 100, 1),
        "probabilities": {k: round(v * 100, 1) for k, v in probabilities.items()},
        "matched_vocab": matched_vocab,
        "char_count":    len(text),
    }