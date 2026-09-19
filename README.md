# Spam/Scam Message Classifier

A machine learning web app that classifies SMS/text messages as spam or legitimate in real time.

## Overview
Trained on the UCI SMS Spam Collection dataset (5,572 labeled messages). Compared three models (Naive Bayes, Logistic Regression, Linear SVM) and selected the best performer based on F1 score.

## Model Performance
- **Model:** Linear SVM with TF-IDF features
- **Precision:** 96.4%
- **Recall:** 90.0%
- **F1 Score:** 0.93

## Tech Stack
- Python, scikit-learn, pandas
- TF-IDF vectorization (3,000 features)
- Streamlit for the web interface

## How It Works
1. Text preprocessing: lowercasing, removing punctuation/numbers, whitespace normalization
2. TF-IDF vectorization to convert text into numerical features
3. Linear SVM classification
4. Confidence score shown via decision boundary distance

## Run Locally

Run "pip install -r requirements.txt" in terminal to install and project runs identically on their machine.