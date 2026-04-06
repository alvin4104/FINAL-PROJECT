# Sentiment Analysis of Qatar Airways Passenger Reviews

**Course:** Computer Linguistics — HUFLIT  
**Project Type:** Sentiment Analysis (Text Classification)  
**Data Source:** Airline Quality Review Platform (airlinequality.com)  
**Labels:** POSITIVE / NEUTRAL / NEGATIVE  

---

## Project Overview

This project performs sentiment analysis on 1,999 English-language passenger reviews of Qatar Airways collected between March 2015 and March 2024. Reviews are classified into three sentiment categories using multiple machine learning models and a VADER lexicon baseline, with linguistic error analysis of misclassified samples.

## Models Used

| Model | Type |
|-------|------|
| Naive Bayes | Machine Learning |
| Support Vector Machine (SVM) | Machine Learning |
| Logistic Regression | Machine Learning |
| Rule-Based Classifier | Baseline |
| VADER | Lexicon-Based Baseline |

## Results Summary

| Model | Accuracy | Weighted F1 |
|-------|----------|-------------|
| **SVM** | **0.770** | **0.740** |
| Logistic Regression | 0.760 | 0.690 |
| Rule-Based | 0.744 | 0.685 |
| Naive Bayes | 0.730 | 0.640 |
| VADER (Lexicon) | 0.578 | 0.550 |

## Pipeline

```
Raw Qatar Airways Reviews (CSV)
      ↓
Text Preprocessing (lowercase, remove digits/punctuation, stop words)
      ↓
Sentiment Labelling (rating 7-10: positive, 4-6: neutral, 1-3: negative)
      ↓
TF-IDF Vectorization (max 3,000 features)
      ↓
80/20 Stratified Train/Test Split (random_state=42)
      ↓
Model Training & Evaluation (Accuracy, Precision, Recall, F1)
      ↓
Visualization + Error Analysis
```

## Dataset

| Attribute | Value |
|-----------|-------|
| Source | airlinequality.com (via Kaggle) |
| Total Reviews | 1,999 (after preprocessing) |
| Language | English |
| Date Range | March 2015 — March 2024 |
| Rating Scale | 1–10 |
| Mean Rating | 7.32 / 10.00 |
| Mean Review Length | 131.4 words |
| Features | 15 columns (text, rating, seat type, traveller type, route, etc.) |

## Sentiment Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Positive (7–10) | 1,376 | 68.8% |
| Negative (1–3) | 326 | 16.3% |
| Neutral (4–6) | 297 | 14.9% |

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/alvin4104/FINAL-PROJECT.git
cd FINAL-PROJECT

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py
```

## Output Files

| File | Description |
|------|-------------|
| `results.png` | Sentiment distribution, model accuracy, WordCloud |
| `confusion_matrix.png` | Confusion matrix of best model (SVM) |

## Install Dependencies

```bash
pip install pandas numpy scikit-learn vaderSentiment wordcloud matplotlib seaborn
```

## Key Findings

- **SVM** achieved the best accuracy (77.0%) confirming its strength in high-dimensional TF-IDF space
- **Business Class** reviews had higher positive sentiment (73.9%) vs Economy Class (65.9%)
- **Neutral class** was hardest to classify (SVM F1 = 0.30) due to class imbalance
- Main error types: mixed-sentiment discourse, negation, implicit sentiment, class boundary ambiguity
