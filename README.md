# Anomaly Detection for Financial Data

## Overview

This project was developed as part of our **Intro to Data Analytics** course. The aim is to detect anomalies in financial data using statistical and machine learning techniques. We used a structured dataset provided via [Kaggle](https://www.kaggle.com/competitions/anomaly-dectection-for-financial/data), which contains both training and testing datasets representing a variety of financial and operational metrics across companies and funds.

---

## Dataset

### 📁 Train Dataset:
- **Shape:** 60 rows × 25 columns
- **Contents:**
  - `company`, `fund` (categorical)
  - Metrics: `IRR`, `W_Avg`, `Reduction`, `Vision`, `Opt_Pw`, `KPIs`, `Bus_Int`, `Out_Exp`, `Cus_Exp`, `FA_Opt`, `Digi_Trans`, `Del_Acct`, etc.
  - `Anomaly_label`: Binary target column (0 = normal, 1 = anomaly)

### 📁 Test Dataset:
- **Shape:** 40 rows × 24 columns
- Similar structure as training data, without `Anomaly_label`

---

## Objectives

- Clean and preprocess the dataset
- Visualize data distribution and understand feature relationships
- Handle missing values and outliers
- Build models to detect anomalies in financial data
- Evaluate model performance using metrics like accuracy, precision, recall, and F1-score

---

## Preprocessing & Cleaning

- Converted `time` column to datetime format
- Standardized column names by removing trailing/leading spaces
- Identified and addressed missing values (`Lead_Own`, `Lead_Own_old`)
- Removed or imputed invalid entries (e.g., negative or zero IRR)

---

## Methodology

- **Exploratory Data Analysis (EDA)** using Pandas, Matplotlib, and Seaborn
- **Outlier Detection**:
  - Statistical thresholding
  - Visual inspections via boxplots
- **Modeling Techniques**:
  - Isolation Forest
  - One-Class SVM
  - Autoencoders (optional)
- **Evaluation**:
  - Compared predictions against labeled `Anomaly_label` in train data
  - Analyzed class imbalance and adjusted using resampling techniques (if necessary)

---

## Tools & Libraries

- **Python 3.9**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **scikit-learn**

---

## Results

- Detected 19 anomalies from the training set
- Achieved high precision in anomaly classification
- Visualizations show clear separability between normal and anomalous points in reduced-dimensional plots

---


## Future Work

- Experiment with ensemble models and deep learning (e.g., Autoencoders)
- Deploy model for real-time anomaly detection
- Explore temporal patterns using time series modeling techniques

