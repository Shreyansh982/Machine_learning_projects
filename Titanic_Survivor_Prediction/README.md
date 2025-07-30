# Titanic Survivor Prediction

A comprehensive machine learning project for predicting passenger survival on the Titanic using advanced feature engineering, multiple classification algorithms, and sophisticated model evaluation techniques.

## 📋 Project Overview

This project implements a binary classification system to predict whether a passenger survived the Titanic disaster based on various passenger characteristics. It demonstrates advanced machine learning techniques including feature engineering, class balancing, hyperparameter tuning, and comprehensive model evaluation.

## 🎯 Problem Statement

The Titanic survival prediction aims to:
- Predict passenger survival (0: Died, 1: Survived) based on passenger data
- Implement sophisticated feature engineering techniques
- Handle class imbalance with advanced balancing methods
- Compare multiple classification algorithms with hyperparameter optimization
- Provide comprehensive model evaluation and feature importance analysis

## 📊 Dataset

- **Dataset**: Titanic passenger dataset (Titanic.csv)
- **Target Variable**: Survival (0: Died, 1: Survived)
- **Features**: Passenger characteristics including:
  - Passenger ID
  - Survival status
  - Passenger class (1st, 2nd, 3rd)
  - Name
  - Sex
  - Age
  - Number of siblings/spouses aboard
  - Number of parents/children aboard
  - Ticket number
  - Fare
  - Cabin
  - Port of embarkation
- **Size**: 1,311 samples with multiple features

## 🛠️ Implementation Details

### Advanced Feature Engineering

The project implements sophisticated feature engineering:

1. **Family Features**:
   - `FamilySize`: Total family members (SibSp + Parch + 1)
   - `IsAlone`: Binary indicator for passengers traveling alone

2. **Fare Features**:
   - `FareBin`: Quartile-based fare categorization
   - Fare discretization for better pattern recognition

3. **Data Cleaning**:
   - Column name standardization
   - Removal of unnecessary columns
   - Missing value handling strategies

### Data Preprocessing

- **Separate Pipelines**: Different preprocessing for different model types
- **Numerical Features**: Age, Fare, SibSp, Parch, FamilySize
- **Categorical Features**: Sex, Embarked, Pclass, IsAlone, FareBin
- **Missing Value Handling**:
  - Median imputation for numerical features
  - Most frequent imputation for categorical features
  - Constant imputation (-1) for tree-based models

### Models Implemented

1. **Logistic Regression**
   - Class balancing with `class_weight='balanced'`
   - Maximum iterations: 200
   - Standardized features with scaling

2. **Support Vector Machine (SVM)**
   - RBF kernel with probability estimation
   - Class balancing for imbalanced datasets
   - Standardized features

3. **Random Forest**
   - 100 estimators with max_depth=5
   - Class balancing and feature importance analysis
   - Hyperparameter tuning with GridSearchCV

4. **XGBoost**
   - Gradient boosting with scale_pos_weight for imbalance
   - Learning rate: 0.1, max depth: 4
   - Comprehensive hyperparameter optimization

### Hyperparameter Tuning

- **Random Forest Tuning**:
  - n_estimators: [50, 100, 200]
  - max_depth: [3, 5, 7]
  - min_samples_split: [2, 5, 10]
  - min_samples_leaf: [1, 2, 4]

- **XGBoost Tuning**:
  - learning_rate: [0.01, 0.1, 0.3]
  - max_depth: [3, 4, 5]
  - n_estimators: [50, 100, 200]
  - subsample: [0.7, 0.9]
  - colsample_bytree: [0.7, 0.9]

### Model Evaluation

- **Cross-validation**: 5-fold cross-validation for robust evaluation
- **Metrics**: Accuracy, Classification Report, Confusion Matrix
- **Feature Importance**: Random Forest-based feature ranking
- **Model Comparison**: Systematic evaluation of all algorithms

## 📈 Results

### Performance Metrics
The project provides comprehensive evaluation:
- **Cross-validation Accuracy**: Mean and standard deviation
- **Validation Accuracy**: Performance on held-out validation set
- **Classification Reports**: Precision, recall, F1-score for each class
- **Confusion Matrices**: Visual representation of predictions

### Model Rankings
Results include:
- **Base Models**: Initial performance of all algorithms
- **Tuned Models**: Performance after hyperparameter optimization
- **Feature Importance**: Most influential features for survival prediction

## 🚀 Usage

1. **Prerequisites**: Install required dependencies from `requirements.txt`
2. **Data**: Ensure `Titanic.csv` is in the `data/` directory
3. **Run**: Execute the main script:
   ```bash
   python Survivor_Prediction.py
   ```

## 📊 Visualizations

### 1. Confusion Matrices
- **Individual Plots**: One for each model
- **Color Schemes**: Blues theme for all matrices
- **Labels**: "Died" and "Survived" for clarity
- **Annotations**: Numerical values in each cell

### 2. Feature Importance Plot
- **Bar Chart**: Horizontal bar chart of feature importance
- **Ranked Features**: Ordered by importance score
- **Color Palette**: Viridis color scheme
- **Comprehensive View**: All engineered and original features

### 3. Model Comparison Plot
- **Accuracy Comparison**: Bar chart of all model accuracies
- **Color Coding**: Viridis palette for visual appeal
- **Rotation**: 15-degree rotation for better readability
- **Y-axis**: Limited to [0, 1] for clear comparison

## 🔧 Key Features

- **Pipeline Architecture**: Modular preprocessing and modeling pipelines
- **Class Imbalance Handling**: Multiple balancing techniques
- **Hyperparameter Optimization**: Automated tuning using GridSearchCV
- **Feature Engineering**: Domain-specific feature creation
- **Comprehensive Evaluation**: Multiple metrics and visualization techniques

## 📝 Technical Notes

- **Preprocessing Pipelines**: Separate pipelines for different model types
- **Memory Efficiency**: Optimized data handling for large datasets
- **Reproducibility**: Fixed random seeds for consistent results
- **Scalability**: Support for parallel processing with n_jobs parameter

## 🎯 Dataset Characteristics

The Titanic dataset is well-known for:
- **Historical Significance**: Real disaster data with survival outcomes
- **Class Imbalance**: More deaths than survivals
- **Mixed Feature Types**: Numerical and categorical features
- **Missing Values**: Requires sophisticated imputation strategies
- **Feature Interactions**: Complex relationships between passenger characteristics

## 📊 Model Insights

### Best Performing Model
The project automatically identifies the best performing model through:
- **Cross-validation Scores**: Robust performance estimation
- **Validation Accuracy**: Performance on unseen data
- **Hyperparameter Optimization**: Fine-tuned model parameters

### Feature Importance Insights
- **Sex**: Typically the most important feature (women and children first)
- **Passenger Class**: Higher classes had better survival rates
- **Family Size**: Optimal family size for survival
- **Fare**: Indicator of socioeconomic status and survival probability

## 🎯 Historical Context

This project has educational value in understanding:
- **Historical Events**: Titanic disaster analysis
- **Social Factors**: Class, gender, and family size effects
- **Data Science**: Real-world application of ML techniques
- **Feature Engineering**: Domain knowledge application

## 📚 Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- imbalanced-learn

## 👨‍💻 Author

This project demonstrates advanced machine learning techniques applied to a classic historical dataset, showcasing comprehensive feature engineering, model selection, and evaluation methods.
