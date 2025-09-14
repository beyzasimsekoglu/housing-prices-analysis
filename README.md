# Housing Prices - Advanced Regression Techniques

## CS5785 Assignment 1 - Part I

This project implements a complete machine learning pipeline for predicting house prices using regression techniques.

## Features

- **Feature Analysis**: Identifies 25 continuous and 55 categorical features
- **Data Preprocessing**: Handles missing values, feature scaling, one-hot encoding
- **OLS Implementation**: Ordinary Least Squares implemented from scratch with ridge regularization
- **Model Evaluation**: Achieves R² = 0.7831 and RMSE = 36,984
- **Kaggle Submission**: Ready-to-submit CSV file

## Files

- `housing_analysis.py` - Complete analysis script
- `feature_distributions.png` - Feature distribution visualizations
- `one_hot_encoding_demo.png` - One-hot encoding demonstration
- `submission.csv` - Kaggle submission file
- `train.csv` - Training data
- `test.csv` - Test data
- `data_description.txt` - Feature descriptions

## Usage

```bash
python3 housing_analysis.py
```

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Results

- **Model Performance**: R² = 0.7831, RMSE = 36,984
- **Features Used**: 11 important features after selection
- **Preprocessing**: Missing value imputation, feature scaling, one-hot encoding
- **Regularization**: Ridge regression for numerical stability

## Key Insights

1. **Continuous Features**: LotArea, YearBuilt, TotalBsmtSF, GrLivArea are most important
2. **Categorical Features**: OverallQual, MSZoning, Neighborhood significantly impact price
3. **One-Hot Encoding**: Created 109 binary features from 17 categorical features
4. **Feature Selection**: Reduced from 80+ features to 11 to prevent overfitting
