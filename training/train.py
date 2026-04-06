"""
train.py - Japanese Sentence Difficulty Classifier
COSC 402 - AI/ML Implementation

This script:
1. Loads the JLPT vocabulary dataset
2. Generates synthetic labeled sentences by combining words
3. Extracts linguistic features (kanji ratio, JLPT score, length, etc.)
4. Trains a scikit-learn RandomForestClassifier
5. Also trains a small TensorFlow/Keras neural network
6. Saves the best model and feature scaler to /models/
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import unicodedata

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ─────────────────────────────────────────────
# 1. LOAD VOCABULARY DATASET
# ─────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "jlpt_vocab.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading JLPT vocabulary dataset...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Kanji", "Level"])
df["Kanji"] = df["Kanji"].astype(str).str.strip()
df["Level"] = df["Level"].astype(int)

print(f"  Loaded {len(df)} vocabulary entries")
print(f"  Level distribution:\n{df['Level'].value_counts().sort_index()}\n")

# ─────────────────────────────────────────────
# 2. BUILD WORD-TO-LEVEL LOOKUP
# ─────────────────────────────────────────────

# Map each word to its JLPT level (1=hardest, 5=easiest)
word_to_level = {}
for _, row in df.iterrows():
    word = row["Kanji"]
    level = row["Level"]
    # Keep the hardest (lowest) level if word appears multiple times
    if word not in word_to_level or level < word_to_level[word]:
        word_to_level[word] = level

print(f"Built lookup for {len(word_to_level)} unique words\n")

# ─────────────────────────────────────────────
# 3. FEATURE EXTRACTION FUNCTIONS
# ─────────────────────────────────────────────

def is_kanji(char):
    """Check if a character is a kanji (CJK Unified Ideograph)."""
    return '\u4e00' <= char <= '\u9fff'

def is_hiragana(char):
    return '\u3041' <= char <= '\u309f'

def is_katakana(char):
    return '\u30a0' <= char <= '\u30ff'

def extract_features(text, word_to_level_map):
    """
    Extract numerical features from a Japanese text string.
    Returns a feature vector suitable for ML models.
    """
    text = str(text).strip()
    if not text:
        return [0] * 7

    total_chars = len(text)
    kanji_chars = sum(1 for c in text if is_kanji(c))
    hiragana_chars = sum(1 for c in text if is_hiragana(c))
    katakana_chars = sum(1 for c in text if is_katakana(c))

    kanji_ratio = kanji_chars / total_chars if total_chars > 0 else 0
    hiragana_ratio = hiragana_chars / total_chars if total_chars > 0 else 0
    katakana_ratio = katakana_chars / total_chars if total_chars > 0 else 0

    # Match words from our vocabulary against the text
    matched_levels = []
    for word, lvl in word_to_level_map.items():
        if word in text and len(word) > 1:
            matched_levels.append(lvl)

    if matched_levels:
        avg_jlpt_level = np.mean(matched_levels)
        min_jlpt_level = min(matched_levels)  # hardest word found
        num_known_words = len(matched_levels)
    else:
        # No known words found — assume unknown = hardest
        avg_jlpt_level = 1.0
        min_jlpt_level = 1
        num_known_words = 0

    features = [
        kanji_ratio,           # ratio of kanji characters
        hiragana_ratio,        # ratio of hiragana
        katakana_ratio,        # ratio of katakana
        avg_jlpt_level,        # average JLPT level of matched vocab (1-5)
        min_jlpt_level,        # hardest word's JLPT level
        num_known_words,       # how many known vocab words were matched
        total_chars,           # total character count (sentence length)
    ]
    return features

# ─────────────────────────────────────────────
# 4. GENERATE SYNTHETIC LABELED DATASET
# ─────────────────────────────────────────────
# Since we have vocab but not full sentences, we generate training samples
# by grouping words by level and creating short "sentence-like" combinations.

print("Generating synthetic labeled training data...")

LABEL_MAP = {
    "easy": 0,
    "intermediate": 1,
    "hard": 2
}
LABEL_NAMES = ["easy", "intermediate", "hard"]

def level_to_label(jlpt_level):
    """Convert JLPT level (1-5) to difficulty label."""
    if jlpt_level >= 4:      # N5, N4
        return "easy"
    elif jlpt_level == 3:    # N3
        return "intermediate"
    else:                    # N2, N1
        return "hard"

# Group words by difficulty
easy_words    = df[df["Level"] >= 4]["Kanji"].tolist()
medium_words  = df[df["Level"] == 3]["Kanji"].tolist()
hard_words    = df[df["Level"] <= 2]["Kanji"].tolist()

np.random.seed(42)

samples = []
labels  = []

def make_sample(word_pool, n_words=3):
    """Pick n words from pool and join them (simulated sentence)."""
    chosen = np.random.choice(word_pool, size=min(n_words, len(word_pool)), replace=False)
    return "".join(chosen)

N_SAMPLES = 2000  # per class

for _ in range(N_SAMPLES):
    # EASY: mostly N4/N5 words
    n = np.random.randint(2, 5)
    text = make_sample(easy_words, n)
    samples.append(text)
    labels.append(LABEL_MAP["easy"])

for _ in range(N_SAMPLES):
    # INTERMEDIATE: mostly N3 words, some easy/hard mixed in
    words = medium_words.copy()
    np.random.shuffle(words)
    n = np.random.randint(2, 5)
    text = make_sample(medium_words, n)
    samples.append(text)
    labels.append(LABEL_MAP["intermediate"])

for _ in range(N_SAMPLES):
    # HARD: mostly N1/N2 words
    n = np.random.randint(2, 6)
    text = make_sample(hard_words, n)
    samples.append(text)
    labels.append(LABEL_MAP["hard"])

print(f"  Generated {len(samples)} training samples ({N_SAMPLES} per class)\n")

# ─────────────────────────────────────────────
# 5. EXTRACT FEATURES FOR ALL SAMPLES
# ─────────────────────────────────────────────

print("Extracting features...")
X = np.array([extract_features(s, word_to_level) for s in samples])
y = np.array(labels)

print(f"  Feature matrix shape: {X.shape}\n")

# ─────────────────────────────────────────────
# 6. TRAIN/TEST SPLIT & SCALING
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 7. TRAIN RANDOM FOREST (scikit-learn)
# ─────────────────────────────────────────────

print("Training Random Forest classifier (scikit-learn)...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    class_weight="balanced"
)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_preds)

print(f"  Random Forest Accuracy: {rf_acc:.4f}")
print(classification_report(y_test, rf_preds, target_names=LABEL_NAMES))

# ─────────────────────────────────────────────
# 8. TRAIN NEURAL NETWORK (TensorFlow/Keras)
# ─────────────────────────────────────────────

print("Training Neural Network (TensorFlow/Keras)...")
nn_model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(3, activation="softmax")  # 3 classes
])

nn_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = nn_model.fit(
    X_train_scaled, y_train,
    validation_split=0.15,
    epochs=50,
    batch_size=64,
    verbose=0
)

_, nn_acc = nn_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"  Neural Network Accuracy: {nn_acc:.4f}\n")

# ─────────────────────────────────────────────
# 9. SAVE MODELS
# ─────────────────────────────────────────────

print("Saving models...")

# Save Random Forest + scaler
with open(os.path.join(MODEL_DIR, "difficulty_model.pkl"), "wb") as f:
    pickle.dump({"model": rf_model, "scaler": scaler}, f)

# Save Neural Network
nn_model.save(os.path.join(MODEL_DIR, "nn_model.keras"))

# Save word lookup
with open(os.path.join(MODEL_DIR, "word_to_level.pkl"), "wb") as f:
    pickle.dump(word_to_level, f)

print(f"  Saved Random Forest → models/difficulty_model.pkl")
print(f"  Saved Neural Network → models/nn_model.keras")
print(f"  Saved word lookup   → models/word_to_level.pkl")
print("\n✅ Training complete! Run `uvicorn app:app --reload` to start the server.")