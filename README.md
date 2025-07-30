# Machine Learning Projects Collection

A comprehensive collection of machine learning projects covering various prediction tasks and algorithms. This repository contains implementations of popular ML problems with detailed analysis, visualization, and model comparison.

## 📁 Project Structure

```
Machine-Learning/
├── data/                           # Centralized dataset storage
│   ├── BostonHousing.csv          # Boston Housing dataset
│   ├── IRIS.csv                   # Iris flower dataset
│   ├── Titanic.csv                # Titanic survival dataset
│   ├── winequality-red.csv        # Red wine quality dataset
│   ├── train.tsv                  # Training data (TSV format)
│   ├── test.tsv                   # Test data (TSV format)
│   └── valid.tsv                  # Validation data (TSV format)
├── Fake_News_Prediction/          # Fake news detection project
├── Flower_Species_Prediction/     # Iris flower classification
├── House_Price_Prediction/        # Boston housing price prediction
├── Redwine_Quality_Prediction/    # Wine quality assessment
└── Titanic_Survivor_Prediction/   # Titanic survival prediction
```

## 🎯 Projects Overview

### 1. [Fake News Prediction](./Fake_News_Prediction/)
- **Task**: Binary classification for fake news detection
- **Dataset**: News articles with labeled authenticity
- **Models**: Logistic Regression, SVM, Random Forest, XGBoost
- **Features**: Text preprocessing, feature engineering, hyperparameter tuning

### 2. [Flower Species Prediction](./Flower_Species_Prediction/)
- **Task**: Multi-class classification for iris flower species
- **Dataset**: Iris dataset (sepal/petal measurements)
- **Models**: Logistic Regression, Random Forest, SVM
- **Features**: Data visualization, feature scaling, model comparison

### 3. [House Price Prediction](./House_Price_Prediction/)
- **Task**: Regression for predicting house prices
- **Dataset**: Boston Housing dataset
- **Models**: Linear Regression, Random Forest, Gradient Boosting
- **Features**: Robust scaling, pipeline preprocessing, residual analysis

### 4. [Red Wine Quality Prediction](./Redwine_Quality_Prediction/)
- **Task**: Binary classification for wine quality assessment
- **Dataset**: Red wine physicochemical properties
- **Models**: Logistic Regression, SVM, Random Forest
- **Features**: Advanced feature engineering, SMOTE balancing, feature selection

### 5. [Titanic Survivor Prediction](./Titanic_Survivor_Prediction/)
- **Task**: Binary classification for passenger survival prediction
- **Dataset**: Titanic passenger data
- **Models**: Logistic Regression, SVM, Random Forest, XGBoost
- **Features**: Feature engineering, class balancing, hyperparameter optimization

## 🛠️ Technologies Used

- **Python 3.8+**
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **XGBoost**: Gradient boosting framework
- **Imbalanced-learn**: Handling imbalanced datasets

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Machine-Learning
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

Each project can be run independently. Navigate to the specific project directory and run the main Python file:

```bash
# Example: Run Titanic survival prediction
cd Titanic_Survivor_Prediction
python Survivor_Prediction.py

# Example: Run Iris flower classification
cd Flower_Species_Prediction
python Species_Prediction.py
```

## 📊 Key Features

- **Comprehensive Model Comparison**: Each project compares multiple ML algorithms
- **Data Visualization**: Rich visualizations including confusion matrices, feature importance, and performance plots
- **Feature Engineering**: Advanced feature creation and selection techniques
- **Hyperparameter Tuning**: Grid search optimization for model parameters
- **Cross-validation**: Robust model evaluation using k-fold cross-validation
- **Handling Imbalanced Data**: SMOTE and class weighting techniques

## 📈 Performance Highlights

- **Iris Classification**: Achieves near-perfect accuracy with multiple algorithms
- **Titanic Survival**: Advanced feature engineering improves prediction accuracy
- **Wine Quality**: Feature selection and balancing techniques enhance model performance
- **House Pricing**: Robust preprocessing handles outliers effectively
- **Fake News**: Multi-algorithm approach with hyperparameter optimization

## 🤝 Contributing

Feel free to contribute to this project by:
- Adding new machine learning projects
- Improving existing implementations
- Adding new algorithms or techniques
- Enhancing documentation

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

Created with ❤️ for machine learning enthusiasts and practitioners.

---

**Note**: Each project directory contains its own detailed README with specific implementation details, results, and usage instructions. 