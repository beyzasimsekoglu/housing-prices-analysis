#!/usr/bin/env python3
"""
CS5785 Assignment 1 - Housing Prices Analysis
Part I: The Housing Prices - Advanced Regression Techniques

This script implements:
1. Feature analysis (continuous and categorical)
2. Data preprocessing
3. One-hot encoding
4. Ordinary Least Squares (OLS) from scratch
5. Model evaluation and Kaggle submission
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class HousingAnalysis:
    def __init__(self, train_path='train.csv', test_path='test.csv'):
        """Initialize the housing analysis class."""
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.continuous_features = []
        self.categorical_features = []
        self.encoded_features = []
        
    def analyze_features(self):
        """Analyze and categorize features as continuous or categorical."""
        print("=" * 60)
        print("FEATURE ANALYSIS")
        print("=" * 60)
        
        # Identify continuous features (numeric with many unique values)
        numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop('Id')  # Remove ID column
        
        for col in numeric_cols:
            unique_vals = self.train_df[col].nunique()
            if unique_vals > 10:  # Threshold for continuous features
                self.continuous_features.append(col)
            else:
                self.categorical_features.append(col)
        
        # Identify categorical features (object type or numeric with few unique values)
        object_cols = self.train_df.select_dtypes(include=['object']).columns
        self.categorical_features.extend(object_cols)
        
        print(f"Continuous features ({len(self.continuous_features)}):")
        for i, feature in enumerate(self.continuous_features[:10], 1):  # Show first 10
            print(f"  {i}. {feature}")
        if len(self.continuous_features) > 10:
            print(f"  ... and {len(self.continuous_features) - 10} more")
        
        print(f"\nCategorical features ({len(self.categorical_features)}):")
        for i, feature in enumerate(self.categorical_features[:10], 1):  # Show first 10
            print(f"  {i}. {feature}")
        if len(self.categorical_features) > 10:
            print(f"  ... and {len(self.categorical_features) - 10} more")
        
        return self.continuous_features, self.categorical_features
    
    def plot_feature_distributions(self):
        """Plot histograms for selected continuous and categorical features."""
        print("\n" + "=" * 60)
        print("FEATURE DISTRIBUTION PLOTS")
        print("=" * 60)
        
        # Select one continuous and one categorical feature for plotting
        continuous_feature = 'LotArea'  # Good continuous feature
        categorical_feature = 'OverallQual'  # Good categorical feature
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot continuous feature
        axes[0].hist(self.train_df[continuous_feature].dropna(), bins=50, alpha=0.7, color='skyblue')
        axes[0].set_title(f'Distribution of {continuous_feature} (Continuous)')
        axes[0].set_xlabel(continuous_feature)
        axes[0].set_ylabel('Frequency')
        
        # Plot categorical feature
        categorical_counts = self.train_df[categorical_feature].value_counts().sort_index()
        axes[1].bar(categorical_counts.index, categorical_counts.values, alpha=0.7, color='lightcoral')
        axes[1].set_title(f'Distribution of {categorical_feature} (Categorical)')
        axes[1].set_xlabel(categorical_feature)
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Plotted distributions for:")
        print(f"  - Continuous: {continuous_feature}")
        print(f"  - Categorical: {categorical_feature}")
        
        return continuous_feature, categorical_feature
    
    def preprocess_data(self):
        """Preprocess the data for modeling."""
        print("\n" + "=" * 60)
        print("DATA PREPROCESSING")
        print("=" * 60)
        
        # Combine train and test for consistent preprocessing
        combined_df = pd.concat([self.train_df.drop('SalePrice', axis=1), self.test_df], ignore_index=True)
        
        print("Preprocessing steps:")
        print("1. Handling missing values...")
        
        # Handle missing values
        # For continuous features, fill with median
        for col in self.continuous_features:
            if col in combined_df.columns:
                combined_df[col].fillna(combined_df[col].median(), inplace=True)
        
        # For categorical features, fill with mode or 'Unknown'
        for col in self.categorical_features:
            if col in combined_df.columns:
                if combined_df[col].isnull().sum() > 0:
                    mode_val = combined_df[col].mode()[0] if not combined_df[col].mode().empty else 'Unknown'
                    combined_df[col].fillna(mode_val, inplace=True)
        
        print("2. Feature selection and encoding...")
        
        # Select important features to avoid overfitting
        important_continuous = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
                               'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
                               'GrLivArea', 'GarageCars', 'GarageArea']
        
        important_categorical = ['MSZoning', 'LotShape', 'LandContour', 'Neighborhood',
                               'BldgType', 'HouseStyle', 'ExterQual', 'ExterCond',
                               'Foundation', 'Heating', 'CentralAir', 'KitchenQual',
                               'Functional', 'GarageType', 'PavedDrive', 'SaleType', 'SaleCondition']
        
        # Filter to only important features
        selected_features = important_continuous + important_categorical
        available_features = [col for col in selected_features if col in combined_df.columns]
        combined_df = combined_df[available_features + ['Id']]
        
        print(f"   - Selected {len(available_features)} important features")
        
        # One-hot encode categorical variables
        categorical_to_encode = [col for col in important_categorical if col in combined_df.columns]
        encoded_df = pd.get_dummies(combined_df, columns=categorical_to_encode, prefix=categorical_to_encode)
        self.encoded_features = [col for col in encoded_df.columns if col not in combined_df.columns]
        
        print(f"   - Encoded {len(categorical_to_encode)} categorical features")
        print(f"   - Created {len(self.encoded_features)} new binary features")
        
        # Select numeric features for modeling
        numeric_features = encoded_df.select_dtypes(include=[np.number]).columns
        numeric_features = numeric_features.drop('Id', errors='ignore')
        
        # Split back into train and test
        train_size = len(self.train_df)
        train_processed = encoded_df[:train_size].copy()
        test_processed = encoded_df[train_size:].copy()
        
        # Add target variable back to training data
        train_processed['SalePrice'] = self.train_df['SalePrice'].values
        
        print("3. Feature scaling...")
        # Scale features
        scaler = StandardScaler()
        feature_cols = [col for col in numeric_features if col != 'SalePrice']
        
        train_processed[feature_cols] = scaler.fit_transform(train_processed[feature_cols])
        test_processed[feature_cols] = scaler.transform(test_processed[feature_cols])
        
        print(f"   - Scaled {len(feature_cols)} features")
        
        self.train_processed = train_processed
        self.test_processed = test_processed
        self.feature_cols = feature_cols
        
        return train_processed, test_processed, feature_cols
    
    def demonstrate_one_hot_encoding(self):
        """Demonstrate one-hot encoding with visualization."""
        print("\n" + "=" * 60)
        print("ONE-HOT ENCODING DEMONSTRATION")
        print("=" * 60)
        
        # Select a categorical feature for demonstration
        demo_feature = 'OverallQual'
        
        print(f"Demonstrating one-hot encoding for: {demo_feature}")
        print(f"Original feature has {self.train_df[demo_feature].nunique()} unique values:")
        print(self.train_df[demo_feature].value_counts().sort_index())
        
        # Create one-hot encoded version
        ohe_demo = pd.get_dummies(self.train_df[demo_feature], prefix=demo_feature)
        
        print(f"\nAfter one-hot encoding, we have {ohe_demo.shape[1]} binary features:")
        print(ohe_demo.columns.tolist())
        
        # Visualize the transformation
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original distribution
        original_counts = self.train_df[demo_feature].value_counts().sort_index()
        axes[0].bar(original_counts.index, original_counts.values, alpha=0.7, color='lightblue')
        axes[0].set_title(f'Original {demo_feature} Distribution')
        axes[0].set_xlabel(demo_feature)
        axes[0].set_ylabel('Count')
        
        # One-hot encoded distribution (sum of each binary feature)
        ohe_counts = ohe_demo.sum().sort_index()
        axes[1].bar(range(len(ohe_counts)), ohe_counts.values, alpha=0.7, color='lightcoral')
        axes[1].set_title(f'One-Hot Encoded {demo_feature} Distribution')
        axes[1].set_xlabel('Binary Features')
        axes[1].set_ylabel('Count')
        axes[1].set_xticks(range(len(ohe_counts)))
        axes[1].set_xticklabels([col.replace(f'{demo_feature}_', '') for col in ohe_counts.index], rotation=45)
        
        plt.tight_layout()
        plt.savefig('one_hot_encoding_demo.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return ohe_demo
    
    def implement_ols_from_scratch(self, X, y):
        """Implement Ordinary Least Squares from scratch."""
        print("\n" + "=" * 60)
        print("ORDINARY LEAST SQUARES IMPLEMENTATION")
        print("=" * 60)
        
        # Add bias term (intercept)
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # OLS formula: β = (X^T X)^(-1) X^T y
        # Use ridge regression for numerical stability
        print("Using ridge regression for numerical stability...")
        
        # Calculate (X^T X) + λI
        XTX = np.dot(X_with_bias.T, X_with_bias)
        lambda_reg = 1e-3  # Small regularization parameter
        XTX_reg = XTX + lambda_reg * np.eye(XTX.shape[0])
        
        # Calculate (X^T X + λI)^(-1)
        XTX_inv = np.linalg.inv(XTX_reg)
        
        # Calculate (X^T X + λI)^(-1) X^T
        XTX_inv_XT = np.dot(XTX_inv, X_with_bias.T)
        
        # Calculate coefficients: β = (X^T X + λI)^(-1) X^T y
        coefficients = np.dot(XTX_inv_XT, y)
        
        print("Ridge-regularized OLS implementation successful!")
        print(f"Regularization parameter (λ): {lambda_reg}")
        print(f"Number of coefficients: {len(coefficients)}")
        print(f"Intercept (bias): {coefficients[0]:.4f}")
        print(f"Feature coefficients range: [{coefficients[1:].min():.4f}, {coefficients[1:].max():.4f}]")
        
        return coefficients, X_with_bias
    
    def predict_ols(self, coefficients, X):
        """Make predictions using OLS coefficients."""
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        predictions = np.dot(X_with_bias, coefficients)
        return predictions
    
    def evaluate_model(self, y_true, y_pred):
        """Evaluate model performance."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        print(f"Model Performance:")
        print(f"  MSE: {mse:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.4f}")
        
        return mse, rmse, r2
    
    def create_kaggle_submission(self, coefficients, test_df, feature_cols, filename='submission.csv'):
        """Create Kaggle submission file."""
        print("\n" + "=" * 60)
        print("CREATING KAGGLE SUBMISSION")
        print("=" * 60)
        
        # Make predictions on test set
        X_test = test_df[feature_cols].values
        predictions = self.predict_ols(coefficients, X_test)
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'Id': test_df['Id'],
            'SalePrice': predictions
        })
        
        # Save submission file
        submission.to_csv(filename, index=False)
        print(f"Submission file saved as: {filename}")
        print(f"Predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")
        
        return submission
    
    def run_complete_analysis(self):
        """Run the complete housing prices analysis."""
        print("CS5785 Assignment 1 - Housing Prices Analysis")
        print("=" * 60)
        
        # Step 1: Feature Analysis
        self.analyze_features()
        
        # Step 2: Plot Feature Distributions
        self.plot_feature_distributions()
        
        # Step 3: Data Preprocessing
        train_processed, test_processed, feature_cols = self.preprocess_data()
        
        # Step 4: One-Hot Encoding Demonstration
        self.demonstrate_one_hot_encoding()
        
        # Step 5: Prepare data for OLS
        X = train_processed[feature_cols].values
        y = train_processed['SalePrice'].values
        
        print(f"\nTraining data shape: {X.shape}")
        print(f"Target variable range: [{y.min():.2f}, {y.max():.2f}]")
        
        # Step 6: Implement OLS from scratch
        coefficients, X_with_bias = self.implement_ols_from_scratch(X, y)
        
        # Step 7: Evaluate on training data
        y_pred_train = self.predict_ols(coefficients, X)
        print("\nTraining Set Performance:")
        self.evaluate_model(y, y_pred_train)
        
        # Step 8: Create Kaggle submission
        submission = self.create_kaggle_submission(coefficients, test_processed, feature_cols)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Files generated:")
        print("  - feature_distributions.png")
        print("  - one_hot_encoding_demo.png")
        print("  - submission.csv")
        
        return {
            'coefficients': coefficients,
            'feature_cols': feature_cols,
            'train_processed': train_processed,
            'test_processed': test_processed,
            'submission': submission
        }

if __name__ == "__main__":
    # Run the complete analysis
    analyzer = HousingAnalysis()
    results = analyzer.run_complete_analysis()
