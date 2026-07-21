# VedGrow_ML_02 - 🏠 House Price Prediction using Machine Learning

## 📌 Project Overview

This project builds and compares multiple machine learning regression models to predict house sale prices using the **Ames Housing Dataset**. It demonstrates a complete machine learning workflow, including data preprocessing, feature engineering, model training, evaluation, and visualization.

The objective is to identify the model that provides the most accurate predictions while understanding the factors that most influence house prices.

---

## 🎯 Objectives

* Clean and preprocess housing data
* Handle missing values and outliers
* Perform feature engineering and feature selection
* Train and compare multiple regression models
* Evaluate model performance using standard regression metrics
* Visualize feature importance
* Build a reproducible machine learning pipeline

---

## 📂 Dataset

**Dataset:** Ames Housing Dataset

The dataset contains detailed information about residential properties, including:

* Lot size
* Neighborhood
* Year built
* Overall quality
* Garage information
* Basement details
* Living area
* Bathrooms
* Sale price (Target Variable)

**Target Variable**

* `SalePrice`

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Jupyter Notebook

---

## 📁 Project Structure

```text
House-Price-Prediction/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebook/
│   └── House_Price_Prediction.ipynb
│
├── images/
│   ├── feature_importance.png
│   ├── actual_vs_predicted.png
│   └── residual_plot.png
│
├── requirements.txt
├── README.md
└── report.pdf
```

---

## ⚙️ Workflow

### 1. Data Loading

* Load the Ames Housing dataset
* Explore dataset structure
* Understand feature types

### 2. Data Cleaning

* Handle missing values using imputation
* Remove outliers using the IQR method
* Check for duplicate records

### 3. Feature Engineering

New features created include:

* House Age
* Total Bathrooms

Categorical variables are converted using One-Hot Encoding.

### 4. Feature Selection

Features with stronger correlation to the target variable are selected for training.

### 5. Model Training

The following regression models are implemented:

* Linear Regression
* Ridge Regression
* Random Forest Regressor

### 6. Model Evaluation

Performance is measured using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

### 7. Visualization

The project includes visualizations such as:

* Correlation Heatmap
* Feature Importance Plot
* Actual vs Predicted Prices
* Residual Plot

---

## 🤖 Machine Learning Models

### Linear Regression

A simple baseline model that assumes a linear relationship between features and house prices.

### Ridge Regression

An improved linear model that applies L2 regularization to reduce overfitting.

### Random Forest Regressor

An ensemble learning model that combines multiple decision trees to improve prediction accuracy and handle nonlinear relationships.

---

## 📊 Evaluation Metrics

| Metric   | Description                                                   |
| -------- | ------------------------------------------------------------- |
| MAE      | Average absolute prediction error                             |
| RMSE     | Penalizes larger prediction errors                            |
| R² Score | Measures how well the model explains variance in house prices |

---

## 📈 Results

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Linear Regression | 17081.765 | 25782.925 | 0.821 |
| Ridge Regression | 17096.719 | 25796.452 | 0.821 |
| Random Forest | 14743.198 | 21410.001 | 0.876 |

---

## 🚀 How to Run the Project

### Clone the repository

```bash
git clone https://github.com/UMPGHacks/VedGrow_ML_02.git
```

### Navigate to the project folder

```bash
cd VedGrow_ML_02
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```
House_Price_Prediction/Model.ipynb
```

and run all notebook cells.

---

## 📸 Output Visualizations

The notebook generates:

* Correlation Heatmap
* Feature Importance Chart
* Actual vs Predicted Plot
* Residual Plot

Store the generated figures inside the **images/** directory.

---

## 📌 Future Improvements

* Hyperparameter tuning using GridSearchCV
* Cross-validation
* XGBoost and LightGBM implementation
* Feature scaling optimization
* Model deployment using Streamlit or Flask
* Automated machine learning (AutoML)

---

## 📚 Learning Outcomes

By completing this project, you will gain practical experience in:

* Data preprocessing
* Feature engineering
* Regression algorithms
* Model comparison
* Performance evaluation
* Data visualization
* End-to-end machine learning workflow

---

## 👨‍💻 Author

**Parag Gupta**

Aspiring AI/ML Engineer | Data Science Enthusiast

---

## ⭐ If you found this project useful, consider giving the repository a star!
