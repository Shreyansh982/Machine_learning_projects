# Red Wine Quality Prediction

A comprehensive machine learning project for predicting red wine quality using advanced feature engineering, multiple classification algorithms, and sophisticated data preprocessing techniques.

## 📋 Project Overview

This project implements a binary classification system to predict whether a red wine is of high quality (quality ≥ 6) or low quality (quality < 6) based on physicochemical properties. It demonstrates advanced machine learning techniques including feature engineering, class balancing, and feature selection.

## 🎯 Problem Statement

The red wine quality prediction aims to:
- Classify wines as high or low quality based on physicochemical measurements
- Handle imbalanced dataset with sophisticated balancing techniques
- Implement advanced feature engineering for improved model performance
- Compare multiple classification algorithms with comprehensive evaluation

## 📊 Dataset

- **Dataset**: Red wine quality dataset (winequality-red.csv)
- **Target Variable**: Binary quality classification (0: Low Quality, 1: High Quality)
- **Original Quality Scale**: 3-8 (converted to binary: ≥6 = High Quality)
- **Features**: 11 physicochemical properties:
  - Fixed acidity
  - Volatile acidity
  - Citric acid
  - Residual sugar
  - Chlorides
  - Free sulfur dioxide
  - Total sulfur dioxide
  - Density
  - pH
  - Sulphates
  - Alcohol
- **Size**: 1,599 samples with 11 features

## 🛠️ Implementation Details

### Advanced Feature Engineering

The project implements sophisticated feature engineering:

1. **Ratio Features**:
   - `alcohol_sulfur_ratio`: Alcohol to total sulfur dioxide ratio
   - `acid_ratio`: Fixed to volatile acidity ratio
   - `free_sulfur_ratio`: Free to total sulfur dioxide ratio

2. **Interaction Features**:
   - `density_alcohol`: Density × Alcohol interaction
   - `ph_acid_interaction`: pH × Fixed acidity interaction
   - `alcohol_quality_proxy`: Complex alcohol-sulphate-volatile acidity interaction

3. **Polynomial Features**:
   - Squared terms for important features (alcohol, volatile acidity, sulphates, citric acid)

### Data Preprocessing

- **Feature Selection**: SelectKBest with f_classif (k=15)
- **Scaling**: RobustScaler for outlier-resistant normalization
- **Class Balancing**: SMOTE with k_neighbors=3 for minority class oversampling
- **Stratified Split**: 80-20 train-validation split

### Models Implemented

1. **Logistic Regression**
   - L2 regularization (C=1.0)
   - Maximum iterations: 10,000
   - Balanced class handling

2. **Support Vector Machine (SVM)**
   - RBF kernel with C=10.0
   - Gamma='scale' for automatic parameter selection
   - Optimized for classification performance

3. **Random Forest**
   - 500 estimators with max_depth=10
   - Balanced class weights
   - Feature importance analysis
   - Optimized hyperparameters for wine quality prediction

### Model Evaluation

- **Stratified K-Fold Cross-validation**: 5-fold with stratification
- **Accuracy Score**: Overall classification performance
- **Confusion Matrix**: Visual representation of predictions
- **Feature Importance**: Random Forest-based feature ranking

## 📈 Results

### Performance Comparison
The project provides comprehensive model comparison:
- **Logistic Regression**: Linear classification approach
- **SVM**: Kernel-based classification with RBF
- **Random Forest**: Ensemble tree-based method

### Feature Importance Analysis
- **Top 10 Features**: Ranked by importance score
- **Feature Insights**: Understanding of wine quality factors
- **Visual Representation**: Horizontal bar chart of feature importance

## 🚀 Usage

1. **Prerequisites**: Install required dependencies from `requirements.txt`
2. **Data**: Ensure `winequality-red.csv` is in the `data/` directory
3. **Run**: Execute the main script:
   ```bash
   python Quality_Prediction.py
   ```

## 📊 Visualizations

### 1. Model Performance Comparison
- **Bar Chart**: Visual comparison of model accuracies
- **Color Coding**: Gold highlighting for best performing model
- **Accuracy Labels**: Numerical accuracy values on bars

### 2. Confusion Matrices
- **Three Subplots**: One for each model
- **Color Schemes**: Blues, Reds, and Greens for different models
- **Detailed Metrics**: True/False positive/negative rates

### 3. Feature Importance Plot
- **Horizontal Bar Chart**: Top 10 most important features
- **Ranked Display**: Features ordered by importance
- **Dark Green Theme**: Professional visualization style

## 🔧 Key Features

- **Advanced Feature Engineering**: 6 engineered features + 4 polynomial terms
- **SMOTE Balancing**: Sophisticated handling of class imbalance
- **Feature Selection**: Automatic selection of 15 best features
- **Robust Scaling**: Outlier-resistant preprocessing
- **Comprehensive Evaluation**: Multiple metrics and visualizations

## 📝 Technical Notes

- **Class Imbalance**: Original dataset has imbalanced quality distribution
- **Feature Engineering**: Domain-specific knowledge applied to wine chemistry
- **Hyperparameter Optimization**: Models tuned for wine quality prediction
- **Reproducibility**: Fixed random seeds for consistent results

## 🎯 Dataset Characteristics

The red wine quality dataset is characterized by:
- **Imbalanced Classes**: More low-quality wines than high-quality
- **Physicochemical Features**: Objective measurements of wine properties
- **Quality Assessment**: Subjective quality ratings converted to binary
- **Domain Knowledge**: Wine chemistry understanding required for feature engineering

## 📊 Model Insights

### Best Performing Model
The project automatically identifies and highlights the best performing model with:
- **Performance Ranking**: Models sorted by accuracy
- **Visual Highlighting**: Gold color for best model
- **Detailed Metrics**: Comprehensive performance analysis

### Feature Importance Insights
- **Alcohol Content**: Typically the most important feature
- **Acidity Levels**: Critical for wine quality assessment
- **Sulphur Compounds**: Important for wine preservation and taste
- **Chemical Interactions**: Complex relationships between features

## 🎯 Business Applications

This project has practical applications in:
- **Wine Industry**: Quality control and production optimization
- **Wine Retail**: Automated quality assessment for pricing
- **Wine Making**: Process optimization based on quality factors
- **Consumer Guidance**: Quality prediction for wine selection

## 📚 Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn

## 👨‍💻 Author

This project demonstrates advanced machine learning techniques applied to wine quality prediction, showcasing sophisticated feature engineering and class balancing methods.
