# Flower Species Prediction (Iris Classification)

A machine learning project for classifying iris flower species using multiple classification algorithms and comprehensive data visualization techniques.

## 📋 Project Overview

This project implements a classic machine learning problem - the Iris flower classification. It demonstrates how to build and compare multiple classification models to predict iris species based on sepal and petal measurements.

## 🎯 Problem Statement

The Iris flower classification is a fundamental machine learning problem that aims to:
- Classify iris flowers into three species based on four measurements
- Compare the performance of different classification algorithms
- Demonstrate data visualization and exploratory data analysis
- Showcase model evaluation techniques

## 📊 Dataset

- **Dataset**: Iris dataset (IRIS.csv)
- **Features**: 
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- **Target**: Species classification (setosa, versicolor, virginica)
- **Size**: 150 samples with 4 features

## 🛠️ Implementation Details

### Data Exploration and Visualization
- **Pairplot**: Comprehensive visualization of feature relationships
- **KDE Plots**: Kernel density estimation for each feature by species
- **Data Distribution**: Analysis of feature distributions across species

### Data Preprocessing
- **Train-Test Split**: 80-20 split with stratification
- **Feature Scaling**: StandardScaler for normalization
- **Stratified Sampling**: Ensures balanced representation of all classes

### Models Implemented

1. **Logistic Regression**
   - Multi-class classification with one-vs-rest strategy
   - Maximum iterations: 200
   - L2 regularization by default

2. **Random Forest**
   - 100 decision trees
   - Ensemble learning approach
   - Feature importance analysis

3. **Support Vector Machine (SVM)**
   - RBF kernel with default parameters
   - C=1.0, gamma='scale'
   - Multi-class classification support

### Model Evaluation

- **Accuracy Score**: Overall classification accuracy
- **Classification Report**: Precision, recall, and F1-score for each class
- **Confusion Matrix**: Visual representation of predictions vs actual
- **Cross-validation**: Robust model evaluation

## 📈 Results

### Model Performance Comparison
The project compares three algorithms:
- **Logistic Regression**: Linear classification approach
- **Random Forest**: Ensemble tree-based method
- **SVM**: Kernel-based classification

### Sample Prediction
The project includes a demonstration prediction:
- **Input**: Sample measurements [6.1, 3.0, 4.6, 1.4]
- **Output**: Predictions from all three models
- **Interpretation**: Model agreement analysis

## 🚀 Usage

1. **Prerequisites**: Install required dependencies from `requirements.txt`
2. **Data**: Ensure `IRIS.csv` is in the `data/` directory
3. **Run**: Execute the main script:
   ```bash
   python Species_Prediction.py
   ```

## 📊 Visualizations

The project generates comprehensive visualizations:

### 1. Pairplot
- Shows relationships between all feature pairs
- Color-coded by species
- Diagonal shows KDE plots for each feature

### 2. Confusion Matrices
- **Logistic Regression**: Blue color scheme
- **Random Forest**: Green color scheme  
- **SVM**: Orange color scheme
- Shows true vs predicted labels for each species

### 3. Model Performance
- Clear comparison of accuracy scores
- Visual representation of model rankings

## 🔧 Key Features

- **Comprehensive EDA**: Detailed exploratory data analysis
- **Multiple Algorithms**: Comparison of different ML approaches
- **Stratified Sampling**: Balanced train-test split
- **Feature Scaling**: Proper preprocessing for algorithms requiring normalization
- **Visual Analysis**: Rich visualizations for understanding data and results

## 📝 Technical Notes

- **Reproducibility**: Fixed random seed (42) for consistent results
- **Data Quality**: No missing values in the Iris dataset
- **Feature Scaling**: Applied to all models for fair comparison
- **Multi-class Support**: All models handle three-class classification

## 🎯 Dataset Characteristics

The Iris dataset is well-known for its:
- **Linearly Separable Classes**: Setosa is easily separable
- **Non-linear Boundaries**: Versicolor and virginica have overlapping regions
- **Balanced Classes**: Equal representation of all three species
- **Clean Data**: No missing values or outliers

## 📊 Model Insights

### Best Performing Model
The project automatically identifies the best performing model based on accuracy scores and provides:
- Model name and accuracy
- Detailed performance metrics
- Visual confirmation of results

### Feature Importance
- Analysis of which features contribute most to classification
- Understanding of feature relationships
- Insights into species differentiation

## 🎯 Educational Value

This project serves as an excellent introduction to:
- **Classification Problems**: Multi-class classification techniques
- **Data Visualization**: Comprehensive plotting with seaborn
- **Model Comparison**: Systematic evaluation of different algorithms
- **Machine Learning Workflow**: Complete ML pipeline from data to results

## 📚 Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## 👨‍💻 Author

This project demonstrates fundamental machine learning concepts using the classic Iris dataset, making it perfect for learning classification techniques and data visualization.
