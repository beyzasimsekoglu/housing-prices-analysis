# 🏠 Housing Prices - Advanced Regression Techniques

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CS5785](https://img.shields.io/badge/CS5785-Assignment%201-orange.svg)](https://github.com/beyzasimsekoglu)

> **CS5785 Assignment 1 - Part I**: Complete machine learning pipeline for predicting house prices using regression techniques with OLS implementation from scratch.

## 🎯 Overview

This project implements a comprehensive machine learning solution for the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) Kaggle competition. The solution features a custom implementation of Ordinary Least Squares (OLS) regression with ridge regularization for numerical stability.

## ✨ Key Features

- 🔍 **Comprehensive Feature Analysis**: Identifies and analyzes 25 continuous and 55 categorical features
- 🛠️ **Advanced Data Preprocessing**: Handles missing values, feature scaling, and one-hot encoding
- ⚙️ **Custom OLS Implementation**: Ordinary Least Squares implemented from scratch with ridge regularization
- 📊 **Model Evaluation**: Achieves R² = 0.7831 and RMSE = 36,984 on training data
- 🏆 **Kaggle Ready**: Complete submission file for competition entry
- 📈 **Visualizations**: Feature distribution plots and one-hot encoding demonstrations

## 📁 Project Structure

```
housing-prices-analysis/
├── 📄 housing_analysis.py          # Main analysis script
├── 🖼️ feature_distributions.png    # Feature distribution visualizations
├── 🖼️ one_hot_encoding_demo.png    # One-hot encoding demonstration
├── 📄 submission.csv               # Kaggle submission file
├── 📄 train.csv                    # Training dataset
├── 📄 test.csv                     # Test dataset
├── 📄 data_description.txt         # Feature descriptions
└── 📄 README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Analysis

```bash
# Clone the repository
git clone https://github.com/beyzasimsekoglu/housing-prices-analysis.git
cd housing-prices-analysis

# Run the complete analysis
python3 housing_analysis.py
```

## 📊 Results & Performance

| Metric | Value |
|--------|-------|
| **R² Score** | 0.7831 |
| **RMSE** | 36,984 |
| **Features Used** | 11 (after selection) |
| **Training Samples** | 1,460 |
| **Test Samples** | 1,459 |

## 🔬 Technical Implementation

### Data Preprocessing
- **Missing Value Handling**: Median imputation for continuous features, mode for categorical
- **Feature Scaling**: StandardScaler for numerical stability
- **One-Hot Encoding**: 109 binary features from 17 categorical variables
- **Feature Selection**: Reduced from 80+ features to 11 most important

### OLS Implementation
```python
# Ridge-regularized OLS formula
θ = (X^T X + λI)^(-1) X^T y
```
- **Regularization**: λ = 0.001 for numerical stability
- **Matrix Operations**: Custom implementation using NumPy
- **No External Libraries**: Pure mathematical implementation

### Key Features Used
- **Continuous**: LotArea, YearBuilt, TotalBsmtSF, GrLivArea, GarageArea
- **Categorical**: OverallQual, MSZoning, Neighborhood, ExterQual, KitchenQual

## 📈 Key Insights

1. **🏘️ Location Matters**: Neighborhood significantly impacts house prices
2. **🏗️ Quality Counts**: OverallQual and ExterQual are strong predictors
3. **📏 Size Matters**: LotArea, TotalBsmtSF, and GrLivArea are crucial
4. **🚗 Garage Value**: GarageArea and GarageCars affect pricing
5. **🏠 Age Factor**: YearBuilt and YearRemodAdd influence values

## 🎓 Academic Context

This project was developed for **CS5785 - Machine Learning** at Cornell University, demonstrating:
- Understanding of linear regression theory
- Implementation of algorithms from scratch
- Data preprocessing and feature engineering
- Model evaluation and interpretation
- Professional software development practices

## 📝 Assignment Requirements Met

- ✅ **Feature Analysis**: Continuous vs categorical feature identification
- ✅ **Data Preprocessing**: Missing values, normalization, categorical encoding
- ✅ **One-Hot Encoding**: Custom implementation with visualization
- ✅ **OLS from Scratch**: No external regression libraries used
- ✅ **Model Evaluation**: MSE and R² score calculation
- ✅ **Kaggle Submission**: Complete competition entry

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Beyza Simsekoglu**  
*CS5785 Student*  
[GitHub](https://github.com/beyzasimsekoglu) | [LinkedIn](https://linkedin.com/in/beyzasimsekoglu)

---

⭐ **Star this repository if you found it helpful!**
