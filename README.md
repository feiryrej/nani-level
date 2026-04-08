<div align="center">

# nani-level

AI-Based Japanese Sentence Difficulty Classification Using Ensemble Machine Learning

[**Technical Paper »**](https://drive.google.com/file/d/1xZnKCNjFqYkBxLtdDU1YNwnDbUTTzqNK/view?usp=sharing)

[Report Bug](https://github.com/feiryrej/nani-level/issues)
·
[Request Feature](https://github.com/feiryrej/nani-level/pulls)

</div>

## Demo
<div align="center">
  <img src="https://github.com/user-attachments/assets/b6a141f1-ed09-4660-a0dd-a216cbc2251b" alt="Demo" width="80%">
</div>

## Overview

This project develops a Japanese sentence difficulty classification system using Artificial Intelligence and Machine Learning. Sentences are classified into three difficulty levels： **Easy**, **Intermediate**, and **Hard** based on linguistic features derived from a JLPT vocabulary dataset. The system combines a Random Forest classifier and a feedforward Neural Network through ensemble learning, and exposes predictions through a FastAPI backend paired with a static web interface.

This repository is the code companion to the technical report *"AI-Based Japanese Sentence Difficulty Classification Using Ensemble Machine Learning"* submitted to the Polytechnic University of the Philippines for Current Trends and Topics in Computing (April 2026).

## Background

The Japanese language presents unique challenges for learners due to its integration of three writing scripts: hiragana, katakana, and kanji. The Japanese Language Proficiency Test (JLPT) organizes language difficulty into five levels from N5 (beginner) to N1 (advanced), providing a standardized framework for measuring linguistic complexity. Traditional difficulty assessment approaches rely on manual annotation or rule-based systems, which are time-consuming and inconsistent across diverse sentence structures.

This project applies machine learning to automate difficulty classification, using vocabulary-derived linguistic features to replicate the kind of judgment a proficient reader would make about sentence complexity.

### Key Features

- **Ensemble Classifier**: Combines a Random Forest (scikit-learn) and a feedforward Neural Network (TensorFlow/Keras) with a weighted 60/40 split, achieving **97.25% accuracy** on the test set
- **Seven Linguistic Features**: Kanji ratio, hiragana ratio, katakana ratio, average JLPT level, minimum JLPT level, matched vocabulary count, and character length
- **Synthetic Dataset**: 6,000 balanced training samples (2,000 per class) generated from a JLPT vocabulary list of 8,505 entries
- **REST API**: FastAPI backend exposing a prediction endpoint that returns the difficulty label, per-class probability scores, and matched vocabulary
- **Web Interface**: Static HTML/CSS/JS frontend for real-time sentence classification without requiring programming knowledge

## Difficulty Levels

The five JLPT proficiency levels are mapped into three classification categories:

| Category | JLPT Levels |
|----------|-------------|
| Easy | N4, N5 |
| Intermediate | N3 |
| Hard | N1, N2 |

## Application Snapshots
### Landing Page
<img width="1919" height="909" alt="image" src="https://github.com/user-attachments/assets/1df119c1-db68-4d40-8c1d-0f3ed1485a89" />

### Easy Level
<img width="1919" height="912" alt="image" src="https://github.com/user-attachments/assets/52abbfb4-a64b-4a16-9009-ac2cec1e1011" />

### Intermediate Level
<img width="1919" height="910" alt="image" src="https://github.com/user-attachments/assets/ec5f6b0a-b3cd-439c-adfd-a70b9d4bb685" />

### Hard Level
<img width="1919" height="909" alt="image" src="https://github.com/user-attachments/assets/459a735e-4b93-4353-a7f3-e8eb47e172da" />

## System Architecture

The system follows a client-server architecture.

**Training** (`training/train.py`) loads the JLPT vocabulary dataset, constructs a word lookup table, generates 6,000 synthetic sentence samples, extracts seven features per sample, and trains both the Random Forest and Neural Network models. Trained models are saved to the `models/` directory.

**Backend** (`app.py`) is a FastAPI REST API that accepts Japanese text at its prediction endpoint, extracts the same seven features, runs the ensemble classifier, and returns the difficulty label with confidence scores and matched vocabulary.

**Frontend** (`static/`) is a static HTML/CSS/JS interface that communicates with the API and presents classification results in real time.

## Project Structure

```
nani-level/
├── models/
│   └── difficulty_model.pkl        # Saved Random Forest model
├── static/
│   ├── index.html                  # Web interface
│   ├── style.css
│   └── script.js
├── training/
│   └── train.py                    # Model training script
├── app.py                          # FastAPI backend
└── .gitignore
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | N |
|-------|----------|-----------|--------|----------|---|
| Random Forest | 96.58% | 96.60% | 96.58% | 96.58% | 1,200 |
| Neural Network | 96.67% | 97.10% | 96.67% | 96.66% | 1,200 |
| **Ensemble (RF+NN)** | **97.25%** | **97.28%** | **97.25%** | **97.24%** | 1,200 |

The ensemble model outperforms both individual classifiers by leveraging complementary classification signals from each. Feature importance analysis identified the minimum JLPT level and average JLPT level as the strongest predictors of sentence difficulty, consistent with the linguistic property that harder sentences contain more N1/N2 vocabulary.

## Key Findings

- The weighted ensemble (60% Random Forest, 40% Neural Network) achieves 97.25% accuracy, outperforming either model individually.
- Minimum JLPT level and average JLPT level are the most influential features, confirming that vocabulary level is the primary driver of perceived difficulty.
- Kanji ratio and sentence length are significant secondary features, reflecting the role of script composition and structural complexity.
- The web application correctly classified the test sentence "経済的な観点から見ると、この政策は複雑な問題を引き起こす可能性がある" as **Hard** with a confidence score of 65.1%, with matched vocabulary including 複雑 (N2).

## Dataset

| Field | Details |
|-------|---------|
| Source | JLPT Vocabulary List |
| Entries | 8,505 words across 5 JLPT levels |
| Unique words | 8,138 |
| Training samples | 6,000 synthetic sentences (2,000 per class) |
| Train/test split | 80% / 20% |
| Features | 7 per sample |

## Setup and Usage

### Prerequisites

- Python 3.8+

### Installation

1. **Clone the repository**:

   ```
   git clone https://github.com/feiryrej/nani-level.git
   cd nani-level
   ```

2. **Install dependencies**:

   ```
   pip install fastapi uvicorn scikit-learn tensorflow numpy
   ```

### Train the Model

Run the training script to generate and save the models:

```
python training/train.py
```

This will load the JLPT vocabulary dataset, generate synthetic training data, train both models, and save the Random Forest to `models/difficulty_model.pkl`.

### Run the Application

Start the FastAPI server:

```
uvicorn app:app --reload
```

Then open your browser and navigate to `http://localhost:8000` to use the web interface.

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | REST API backend and prediction endpoint |
| `uvicorn` | ASGI server for running the FastAPI app |
| `scikit-learn` | Random Forest classifier and feature scaling |
| `tensorflow` | Neural Network model (Keras API) |
| `numpy` | Numerical operations and feature vector construction |

## Contributors

<table style="width: 100%; text-align: center;">
  <thead>
    <tr>
      <th>Name</th>
      <th>Avatar</th>
      <th>GitHub</th>
      <th>Contributions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Regina S. Bonifacio</td>
      <td><img src="https://github.com/user-attachments/assets/8caf5539-c233-4cc0-a203-36226d033474" alt="" style="border-radius: 50%; width: 50px;"></td>
      <td><a href="https://github.com/feiryrej">feiryrej</a></td>
      <td><b>Developer & Researcher</b>: Responsible for the full development of this project, including dataset selection, synthetic data generation, feature engineering, model training and evaluation, ensemble design, FastAPI backend development, web interface implementation, technical report writing, and repository documentation.</td>
    </tr>
  </tbody>
</table>

## References

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Chollet, F. (2015). Keras. https://keras.io

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press. https://www.deeplearningbook.org

Japan Foundation. (2023). Japanese-Language Proficiency Test (JLPT): About the JLPT. https://www.jlpt.jp/e/about/levelsummary.html

Jurafsky, D., & Martin, J. H. (2026). *Speech and language processing* (3rd ed.). Stanford University. https://web.stanford.edu/~jurafsky/slp3

[[Back to top](#nani-level)]
