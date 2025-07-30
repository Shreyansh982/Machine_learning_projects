# House Price Prediction

A machine learning project for predicting house prices using the Boston Housing dataset with multiple regression algorithms and comprehensive model evaluation.

## 📋 Project Overview

This project implements a regression problem to predict house prices in Boston using various machine learning algorithms. It demonstrates how to build, compare, and evaluate regression models for real estate price prediction.

## 🎯 Problem Statement

The Boston Housing price prediction aims to:
- Predict median house prices based on various neighborhood features
- Compare the performance of different regression algorithms
- Implement robust preprocessing techniques for handling outliers
- Provide comprehensive model evaluation and residual analysis

## 📊 Dataset

- **Dataset**: Boston Housing dataset (BostonHousing.csv)
- **Target Variable**: `medv` (Median house price in $1000s)
- **Features**: 13 neighborhood characteristics including:
  - Crime rate
  - Residential land zoning
  - Industrial proportion
  - River proximity
  - Air quality
  - Room size
  - Age of housing
  - Accessibility to highways
  - Property tax rate
  - Student-teacher ratio
  - Racial composition
  - Socioeconomic status
- **Size**: 506 samples with 13 features

## 🛠️ Implementation Details

### Data Exploration
- **Dataset Info**: Comprehensive overview of data structure
- **Descriptive Statistics**: Summary statistics for all features
- **Data Quality**: Analysis of missing values and data types

### Data Preprocessing
- **Pipeline Architecture**: Modular preprocessing pipeline
- **Missing Value Handling**: Median imputation strategy
- **Feature Scaling**: RobustScaler for outlier-resistant scaling
- **Train-Test Split**: 80-20 split with random state for reproducibility

### Models Implemented

1. **Linear Regression**
   - Basic linear regression model
   - Assumes linear relationship between features and target
   - Baseline model for comparison

2. **Random Forest Regressor**
   - 100 decision trees
   - Ensemble learning approach
   - Handles non-linear relationships
   - Feature importance analysis

3. **Gradient Boosting Regressor**
   - Sequential ensemble method
   - 100 estimators with default parameters
   - Handles complex non-linear patterns
   - Robust to outliers

### Model Evaluation

- **Mean Absolute Error (MAE)**: Average absolute prediction error
- **Root Mean Square Error (RMSE)**: Square root of mean squared error
- **Residual Analysis**: Visual assessment of model assumptions
- **Model Comparison**: Systematic evaluation of all algorithms

## 📈 Results

### Performance Metrics
The project evaluates models using:
- **MAE**: Measures average absolute deviation from actual prices
- **RMSE**: Penalizes larger errors more heavily
- **Residual Plots**: Visual assessment of model performance

### Model Rankings
Results are presented in a structured format showing:
- Model name and performance metrics
- Comparative analysis of all algorithms
- Best performing model identification

## 🚀 Usage

1. **Prerequisites**: Install required dependencies from `requirements.txt`
2. **Data**: Ensure `BostonHousing.csv` is in the `data/` directory
3. **Run**: Execute the main script:
   ```bash
   python Price_Prediction.py
   ```

## 📊 Visualizations

### Residual Plot
- **Purpose**: Assess model assumptions and identify patterns
- **Features**: 
  - Scatter plot of actual vs predicted values
  - Lowess smoothing line for trend identification
  - Helps identify systematic errors or non-linear patterns

### Model Performance Table
- **Structured Output**: Clear comparison of all models
- **Metrics**: MAE and RMSE for each algorithm
- **Ranking**: Performance-based model ordering

## 🔧 Key Features

- **Robust Preprocessing**: Outlier-resistant scaling with RobustScaler
- **Pipeline Architecture**: Modular and reproducible preprocessing
- **Multiple Algorithms**: Comparison of linear and non-linear approaches
- **Comprehensive Evaluation**: Multiple error metrics and visual analysis
- **Reproducible Results**: Fixed random seeds for consistency

## 📝 Technical Notes

- **RobustScaler**: Chosen for its resistance to outliers in housing data
- **Median Imputation**: Conservative approach for missing values
- **Pipeline Design**: Ensures consistent preprocessing across all models
- **Error Metrics**: Both MAE and RMSE for comprehensive evaluation

## 🎯 Dataset Characteristics

The Boston Housing dataset is well-known for:
- **Real-world Application**: Actual housing market data
- **Mixed Feature Types**: Both continuous and categorical features
- **Outliers**: Presence of extreme values requiring robust methods
- **Non-linear Relationships**: Complex interactions between features

## 📊 Model Insights

### Algorithm Comparison
- **Linear Regression**: Baseline performance for linear relationships
- **Random Forest**: Handles non-linear patterns and feature interactions
- **Gradient Boosting**: Sequential learning for complex patterns

### Feature Importance
- Analysis of which neighborhood characteristics most influence prices
- Understanding of real estate market factors
- Insights for urban planning and investment decisions

## 🎯 Business Applications

This project has practical applications in:
- **Real Estate Valuation**: Automated property price estimation
- **Investment Analysis**: Market trend identification
- **Urban Planning**: Understanding factors affecting housing costs
- **Risk Assessment**: Identifying over/under-valued properties

## 📚 Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## 👨‍💻 Author

This project demonstrates practical regression techniques using real-world housing data, making it valuable for understanding both machine learning concepts and real estate market dynamics.
