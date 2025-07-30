import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Load and preprocess data
df = pd.read_csv("data/Titanic.csv")
df.rename(columns={'2urvived': 'Survived', 'Passengerid': 'PassengerId', 'sibsp': 'SibSp'}, inplace=True)
df = df.loc[:, ~df.columns.str.startswith('zero')]

# Feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False, duplicates='drop')

# Define features
X = df.drop(['Survived', 'PassengerId'], axis=1)
y = df['Survived']
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
categorical_features = ['Sex', 'Embarked', 'Pclass', 'IsAlone', 'FareBin']

# Preprocessing pipelines
# For Logistic Regression and SVM
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
# For Random Forest and XGBoost (no scaling needed)
numeric_transformer_rf_xgb = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=-1))
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessors for different models
preprocessor_lr_svm = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])
preprocessor_rf_xgb = ColumnTransformer([
    ('num', numeric_transformer_rf_xgb, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models with class weighting
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
models = {
    'Logistic Regression': Pipeline([
        ('preprocessor', preprocessor_lr_svm),
        ('classifier', LogisticRegression(max_iter=200, class_weight='balanced'))
    ]),
    'SVM': Pipeline([
        ('preprocessor', preprocessor_lr_svm),
        ('classifier', SVC(probability=True, class_weight='balanced'))
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor_rf_xgb),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('preprocessor', preprocessor_rf_xgb),
        ('classifier', XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=100, 
                                    scale_pos_weight=scale_pos_weight, eval_metric='logloss'))
    ])
}

# Hyperparameter tuning for Random Forest and XGBoost
rf_param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}
xgb_param_grid = {
    'classifier__learning_rate': [0.01, 0.1, 0.3],
    'classifier__max_depth': [3, 4, 5],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__subsample': [0.7, 0.9],
    'classifier__colsample_bytree': [0.7, 0.9]
}

# Evaluate models
accuracies = {}
for name, pipeline in models.items():
    # Cross-validation
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print(f"{name} Cross-Validation Accuracy: {scores.mean():.4f} (±{scores.std():.4f})")
    
    # Train and evaluate on validation set
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    accuracies[name] = acc
    print(f"{name} Validation Accuracy: {acc:.4f}")
    print(f"{name} Classification Report:\n", classification_report(y_val, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Died', 'Survived'], yticklabels=['Died', 'Survived'])
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# Tune Random Forest
rf_grid = GridSearchCV(models['Random Forest'], rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)
print(f"Best Random Forest Params: {rf_grid.best_params_}")
print(f"Best Random Forest Accuracy: {rf_grid.best_score_:.4f}")
accuracies['Random Forest Tuned'] = accuracy_score(y_val, rf_grid.predict(X_val))

# Tune XGBoost
xgb_grid = GridSearchCV(models['XGBoost'], xgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
xgb_grid.fit(X_train, y_train)
print(f"Best XGBoost Params: {xgb_grid.best_params_}")
print(f"Best XGBoost Accuracy: {xgb_grid.best_score_:.4f}")
accuracies['XGBoost Tuned'] = accuracy_score(y_val, xgb_grid.predict(X_val))

# Feature importance for Random Forest
rf_best = rf_grid.best_estimator_
feature_names = (numeric_features + 
                 rf_best.named_steps['preprocessor']
                 .named_transformers_['cat']
                 .named_steps['onehot']
                 .get_feature_names_out(categorical_features).tolist())
importances = rf_best.named_steps['classifier'].feature_importances_
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print("\nRandom Forest Feature Importance:\n", feature_importance)

plt.figure(figsize=(8, 5))
sns.barplot(data=feature_importance, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()

# Plot model accuracies
accuracy_df = pd.DataFrame({'Model': list(accuracies.keys()), 'Accuracy': list(accuracies.values())})
plt.figure(figsize=(8, 5))
sns.barplot(data=accuracy_df, x='Model', y='Accuracy', hue='Model', palette='viridis', legend=False)
plt.title("Model Validation Accuracies")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
