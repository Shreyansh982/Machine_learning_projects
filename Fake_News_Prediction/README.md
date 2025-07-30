# Fake News Prediction

A machine learning project for detecting fake news articles using multiple classification algorithms and advanced feature engineering techniques.

## 📋 Project Overview

This project implements a comprehensive fake news detection system that compares multiple machine learning algorithms to identify the most effective approach for distinguishing between real and fake news articles.

## 🎯 Problem Statement

Fake news detection is a critical challenge in the digital age. This project aims to:
- Classify news articles as either real or fake
- Compare the performance of different ML algorithms
- Implement advanced feature engineering techniques
- Handle class imbalance in the dataset

## 📊 Dataset

- **Source**: News articles dataset with labeled authenticity
- **Features**: Text-based features extracted from news articles
- **Target**: Binary classification (0: Real, 1: Fake)
- **Size**: Multiple articles with various features

## 🛠️ Implementation Details

### Data Preprocessing
- **Feature Engineering**: 
  - Family size calculation
  - Alone passenger identification
  - Fare binning for categorical features
- **Handling Missing Values**: 
  - Median imputation for numerical features
  - Most frequent imputation for categorical features
- **Feature Scaling**: StandardScaler for algorithms requiring normalization

### Models Implemented

1. **Logistic Regression**
   - Class balancing with `class_weight='balanced'`
   - Maximum iterations: 200
   - Preprocessing pipeline with scaling

2. **Support Vector Machine (SVM)**
   - RBF kernel with probability estimation
   - Class balancing for imbalanced datasets
   - Standardized features

3. **Random Forest**
   - 100 estimators with max depth of 5
   - Class balancing and feature importance analysis
   - Hyperparameter tuning with GridSearchCV

4. **XGBoost**
   - Gradient boosting with scale_pos_weight for imbalance
   - Learning rate: 0.1, max depth: 4
   - Comprehensive hyperparameter optimization

### Feature Engineering

The project includes sophisticated feature engineering:
- **FamilySize**: Combined family members (SibSp + Parch + 1)
- **IsAlone**: Binary indicator for passengers traveling alone
- **FareBin**: Quartile-based fare categorization
- **Feature Selection**: Automatic selection of most important features

### Model Evaluation

- **Cross-validation**: 5-fold cross-validation for robust evaluation
- **Metrics**: Accuracy, Classification Report, Confusion Matrix
- **Hyperparameter Tuning**: Grid search for Random Forest and XGBoost
- **Feature Importance**: Analysis using Random Forest feature importances

## 📈 Results

The project provides comprehensive model comparison with:
- Cross-validation accuracy scores
- Validation set performance metrics
- Confusion matrices for each model
- Feature importance rankings
- Hyperparameter optimization results

### Performance Metrics
- **Accuracy**: Model classification accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity of the model
- **F1-Score**: Harmonic mean of precision and recall

## 🚀 Usage

1. **Prerequisites**: Install required dependencies from `requirements.txt`
2. **Data**: Ensure the dataset is in the `data/` directory
3. **Run**: Execute the main script:
   ```bash
   python Fake_News_Prediction.py
   ```

## 📊 Visualizations

The project generates several visualizations:
- **Confusion Matrices**: For each model showing true vs predicted labels
- **Feature Importance Plot**: Bar chart of most important features
- **Model Comparison**: Bar chart comparing validation accuracies
- **Performance Metrics**: Detailed classification reports

## 🔧 Key Features

- **Pipeline Architecture**: Modular preprocessing and modeling pipelines
- **Class Imbalance Handling**: SMOTE and class weighting techniques
- **Hyperparameter Optimization**: Automated tuning using GridSearchCV
- **Feature Selection**: Automatic identification of important features
- **Comprehensive Evaluation**: Multiple metrics and visualization techniques

## 📝 Technical Notes

- **Preprocessing Pipelines**: Separate pipelines for different model types
- **Memory Efficiency**: Optimized data handling for large datasets
- **Reproducibility**: Fixed random seeds for consistent results
- **Scalability**: Support for parallel processing with n_jobs parameter

## 🎯 Future Improvements

Potential enhancements for the project:
- **Deep Learning Models**: Neural network implementations
- **Text Processing**: Advanced NLP techniques for text features
- **Ensemble Methods**: Voting and stacking classifiers
- **Real-time Prediction**: API development for live predictions
- **Model Deployment**: Production-ready model serving

## 📚 Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- imbalanced-learn

## 👨‍💻 Author

This project demonstrates advanced machine learning techniques for fake news detection, showcasing best practices in data preprocessing, model selection, and evaluation.
