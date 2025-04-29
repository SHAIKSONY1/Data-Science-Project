# Data-Science-Project
# Customer Churn Prediction using IBM Telco Dataset

## Overview

This project focuses on predicting customer churn in a subscription-based business using a comprehensive data-driven pipeline. It combines Exploratory Data Analysis (EDA), machine learning, and a user-facing deployment via a Streamlit application to deliver actionable insights for customer retention.

## Dataset

- **Source**: IBM Telco Customer Churn Dataset
- **Size**: 7,043 records
- **Features**: 21 features including customer demographics, service subscriptions, and billing info
- **Target**: `Churn` (Yes/No)

## Objectives

1. Identify key factors influencing customer churn
2. Develop robust machine learning models for churn prediction
3. Optimize model performance on an imbalanced dataset
4. Deploy the best-performing model in a web application for real-time predictions

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Analyzed churn distribution and feature relationships
- Key findings: High churn among short-tenure customers, month-to-month contracts, and customers lacking add-on services

### 2. Data Preprocessing
- **Cleaning**: Handled missing values, corrected data types
- **Encoding**: Applied label and one-hot encoding for categorical features
- **Scaling**: Used Min-Max scaling on numeric features
- **Balancing**: Applied SMOTE to address class imbalance

### 3. Feature Selection
- Used Random Forest to identify top 10 features
- Developed two modeling tracks: Full-feature vs. Reduced-feature models

### 4. Modeling
- Algorithms tested:
  - Logistic Regression
  - Decision Tree
  - LightGBM
  - Bagging with LightGBM
  - Artificial Neural Network (ANN)
  - XGBoost, AdaBoost, CatBoost
- **Best Model**: ANN with top 8 features

### 5. Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score, ROC-AUC, MCC, Cohen’s Kappa, Confusion Matrix
- Focused on Recall and F1 Score due to class imbalance

## Deployment

- **Framework**: Streamlit for frontend, FastAPI for backend
- **Functionality**: Accepts customer attributes, returns churn prediction and probability
- **Output**: Real-time predictions to support CRM actions

## Results

| Model                  | Accuracy | Precision | Recall | F1 Score |
|------------------------|----------|-----------|--------|----------|
| ANN (Top 8 Features)   | 0.7852   | 0.7855    | 0.7852 | 0.7854   |
| Bagging LightGBM       | 0.7665   | 0.8072    | 0.7665 | 0.7773   |
| LightGBM               | 0.7516   | 0.7961    | 0.7516 | 0.7634   |

## Business Impact

- Enables targeted retention campaigns
- Improves customer lifetime value
- Reduces operational costs through focused interventions
- Provides insights for service and pricing improvements

## Future Work

- Integrate time-series data and behavioral logs
- Introduce cost-sensitive learning
- Implement Explainable AI (SHAP, LIME)
- Deploy in production with CI/CD pipelines
- Monitor model drift and automate retraining

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
