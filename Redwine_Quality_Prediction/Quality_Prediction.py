import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/winequality-red.csv')
print(f"Dataset shape: {df.shape}")
print(f"Quality distribution:\n{df['quality'].value_counts().sort_index()}")

df['quality_binary'] = (df['quality'] >= 6).astype(int)
print(f"\nBinary quality distribution:\n{df['quality_binary'].value_counts()}")

X = df.drop(['quality', 'quality_binary'], axis=1)
y = df['quality_binary']

print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

X_enhanced = X.copy()
X_enhanced['alcohol_sulfur_ratio'] = X['alcohol'] / (X['total sulfur dioxide'] + 1)
X_enhanced['acid_ratio'] = X['fixed acidity'] / (X['volatile acidity'] + 0.1)
X_enhanced['free_sulfur_ratio'] = X['free sulfur dioxide'] / (X['total sulfur dioxide'] + 1)
X_enhanced['density_alcohol'] = X['density'] * X['alcohol']
X_enhanced['ph_acid_interaction'] = X['pH'] * X['fixed acidity']
X_enhanced['alcohol_quality_proxy'] = X['alcohol'] * X['sulphates'] / (X['volatile acidity'] + 0.1)

important_features = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid']
for feature in important_features:
    X_enhanced[f'{feature}_squared'] = X[feature] ** 2

print(f"Enhanced feature set shape: {X_enhanced.shape}")

X_train, X_val, y_train, y_val = train_test_split(
    X_enhanced, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "="*50)
print("PREPROCESSING")
print("="*50)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

selector = SelectKBest(score_func=f_classif, k=15)
X_train_selected_scaled = selector.fit_transform(X_train_scaled, y_train)
X_val_selected_scaled = selector.transform(X_val_scaled)

X_train_selected = selector.transform(X_train)
X_val_selected = selector.transform(X_val)

selected_features = X_enhanced.columns[selector.get_support()].tolist()
print(f"Selected features ({len(selected_features)}): {selected_features}")

print(f"\nOriginal class distribution: {np.bincount(y_train)}")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_balanced_scaled, y_train_balanced = smote.fit_resample(X_train_selected_scaled, y_train)
X_train_balanced_unscaled, _ = smote.fit_resample(X_train_selected, y_train)
print(f"After SMOTE: {np.bincount(y_train_balanced)}")

print("\n" + "="*50)
print("MODEL TRAINING")
print("="*50)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Training Logistic Regression...")
lr_model = LogisticRegression(C=1.0, penalty='l2', max_iter=10000, random_state=42)
lr_model.fit(X_train_balanced_scaled, y_train_balanced)
y_pred_lr = lr_model.predict(X_val_selected_scaled)
lr_accuracy = accuracy_score(y_val, y_pred_lr)
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

print("Training SVM...")
svm_model = SVC(C=10.0, gamma='scale', kernel='rbf', random_state=42)
svm_model.fit(X_train_balanced_scaled, y_train_balanced)
y_pred_svm = svm_model.predict(X_val_selected_scaled)
svm_accuracy = accuracy_score(y_val, y_pred_svm)
print(f"SVM Accuracy: {svm_accuracy:.4f}")

print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_balanced_unscaled, y_train_balanced)
y_pred_rf = rf_model.predict(X_val_selected)
rf_accuracy = accuracy_score(y_val, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)

models = ['Logistic Regression', 'SVM', 'Random Forest']
accuracies = [lr_accuracy, svm_accuracy, rf_accuracy]
predictions = [y_pred_lr, y_pred_svm, y_pred_rf]

results = list(zip(models, accuracies, predictions))
results = sorted(results, key=lambda x: x[1], reverse=True)

print(f"{'Model':<20} {'Accuracy':<10}")
print("-" * 30)
for model, acc, _ in results:
    print(f"{model:<20} {acc:<10.4f}")

print(f"\nBest Model: {results[0][0]} with {results[0][1]:.4f} accuracy")

plt.figure(figsize=(10, 6))
colors = ['skyblue', 'lightcoral', 'lightgreen']
bars = plt.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.ylim(0.7, max(accuracies) + 0.03)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

best_idx = accuracies.index(max(accuracies))
bars[best_idx].set_color('gold')
bars[best_idx].set_edgecolor('darkorange')
bars[best_idx].set_linewidth(3)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ['Blues', 'Reds', 'Greens']
for i, (model, acc, pred) in enumerate(zip(models, accuracies, predictions)):
    cm = confusion_matrix(y_val, pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=colors[i], ax=axes[i],
                cbar_kws={'shrink': 0.8},
                xticklabels=['Low Quality', 'High Quality'],
                yticklabels=['Low Quality', 'High Quality'])
    
    axes[i].set_title(f'{model}\nAccuracy: {acc:.4f}', fontweight='bold')
    axes[i].set_xlabel('Predicted', fontweight='bold')
    if i == 0:
        axes[i].set_ylabel('Actual', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("="*50)

feature_importance = rf_model.feature_importances_
feature_df = pd.DataFrame({
    'feature': selected_features,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
for i, (feat, imp) in enumerate(zip(feature_df['feature'][:10], feature_df['importance'][:10])):
    print(f"{i+1:2d}. {feat:<25}: {imp:.4f}")

plt.figure(figsize=(10, 8))
top_features = feature_df.head(10)
plt.barh(range(len(top_features)), top_features['importance'], color='darkgreen', alpha=0.7)
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance', fontweight='bold')
plt.title('Top 10 Feature Importance (Random Forest)', fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
