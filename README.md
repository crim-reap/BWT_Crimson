---

# Stop Fraud Before Money Moves


---

## Problem Statement
Fraudsters use Al and social engineering to trick users.Current systems detect fraud only after money has already moved.

---

## Project Overview
this project aims to intercept and block suspicious transactions within sub-300 milliseconds during the transaction lifecycle. Its a project on cyber threat detection where we are Stopping Fraud Before the Fraudsters use Al and social engineering to trick users. Current systems detect fraud only after money has already moved.

---

## Objective
-Detect fraud in real time (<300ms)

-Prevent fund transfer for high-risk transactions

-Minimize false positives

-Ensure scalable and secure deployment

-Build a proactive fraud interception engine

---
## Architecture Overview

### 1. Data-Driven Model Training

We use the IEEE-CIS Fraud Detection dataset (Kaggle) to train our models.

The dataset includes:
- Transaction-level data (amount, time, card type, billing match)

- Identity-level data (device type, browser, IP patterns, email domain)

- The model learns patterns that differentiate:

  - Normal transactions
  - Fraudulent transactions

### 2. Machine Learning Pipeline
   
Step 1: Data Cleaning

- Handle missing values

- Remove irrelevant columns

Step 2: Feature Engineering

- We create intelligent behavioral signals such as:

- Time since last transaction

- Rapid transaction frequency

Step 3: Handling Imbalanced Data

- Since fraud cases are rare (~3%), we use:

- Class weighting in XGBoost

OR

- SMOTE (Synthetic Minority Oversampling Technique)

This ensures the model does not ignore fraud cases.


Step 4: Risk Scoring & Threshold Optimization

- We tune decision thresholds to:

- Minimize false positives

- Maintain high fraud detection recall



Models are pre-trained and loaded into memory, ensuring ultra-fast prediction.



<img width="2816" height="1504" alt="architecture diagram" src="https://github.com/user-attachments/assets/da4791a4-7726-4b11-af35-a49301572ced" />

---

## Technical Stack
### Programming Language
-Python 3.10+

### Machine Learning
- XGBoost
- Scikit-learn
- Isolation Forest
- Pandas
- NumPy

### Frontend
- HTML
- CSS
- JavaScript
  
### Backend
- FastAPI
- Uvicorn

### AI Development Assistant
- TRAE (used for accelerating model development, pipeline automation, and rapid prototyping)

## Key Features

- Real-time fraud detection
- Sub-300ms response time
- Dual-layer ML detection
- Optimized for low false positives
- Scalable architecture

---

## Future Improvements

- Real-time behavioral profiling
- Continuous model retraining
- Graph-based fraud detection
- Model explainability (SHAP)
- Cloud deployment & scaling
