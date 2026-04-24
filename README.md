# 🎓 Student Performance Prediction (ML Project)

## 📌 Overview

This project is a complete end-to-end Machine Learning pipeline that predicts a student’s math score based on demographic and academic features.

It includes:

- Exploratory Data Analysis (EDA)
- Data preprocessing pipeline
- Multiple regression models
- Model comparison & evaluation
- Modular production-ready code (src/ pipeline)

---

## 📊 Problem Statement

The goal is to predict:

➡️ math_score

using features like:

- Gender
- Race/Ethnicity
- Parental level of education
- Lunch type
- Test preparation course
- Reading score
- Writing score

---

## 🔍 Exploratory Data Analysis (EDA)

The EDA notebook (EDA.ipynb) focuses on:

### Key Insights

- 📈 Reading & writing scores strongly correlate with math score
- 🍽️ Students with standard lunch perform better
- 🎓 Completing test preparation course improves scores
- 👨‍👩‍👧 Parental education has a moderate impact
- ⚖️ Gender differences exist but are not dominant

### Visualizations Included

- Distribution plots
- Correlation heatmap
- Box plots for categorical variables
- Pairplots for feature relationships

---

## ⚙️ Project Architecture

```text
ML_PROJ/
│
├── notebook/
│   ├── EDA.ipynb                  # Data analysis & visualization
│   └── Model.ipynb                # Model training & evaluation
│
├── src/
│   ├── components/
│   │   ├── __init__.py            # Project structure
│   │   ├── data_ingestion.py      # Load & split dataset
│   │   ├── data_transformation.py # Preprocessing pipeline
│   │   └── model_trainer.py       # Model training & evaluation
│   │
│   ├── pipeline/                  # Training / prediction pipeline
│   ├── exception.py               # Custom exception handling
│   ├── logger.py                  # Logging system
│   └── utils.py                   # Helper functions
│
├── artifacts/                     # Saved models & preprocessors
├── requirements.txt
├── setup.py
└── README.md

## 🔄 Machine Learning Pipeline

### 1. Data Ingestion
- Reads dataset
- Splits into train/test

### 2. Data Transformation
- One-hot encoding for categorical features
- Standard scaling for numerical features
- Pipeline built using ColumnTransformer

### 3. Model Training

Multiple regression models are trained:

- Linear Regression
- Ridge Regression
- Lasso Regression
- KNN Regressor
- Decision Tree
- Random Forest
- AdaBoost
- XGBoost
- CatBoost

