import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Load dataset
df = pd.read_csv('data/IRIS.csv')

# Visualize data
sns.pairplot(df, hue='species', diag_kind='kde')
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

# Split into features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Logistic Regression
# -----------------------------
log_model = LogisticRegression(max_iter=200, random_state=42)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

acc_log = accuracy_score(y_test, y_pred_log)
print("\n--- Logistic Regression ---")
print("Accuracy:", acc_log)
cm_log = confusion_matrix(y_test, y_pred_log, labels=log_model.classes_)
disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log, display_labels=log_model.classes_)
disp_log.plot(cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.show()
print(classification_report(y_test, y_pred_log))

# -----------------------------
# Random Forest
# -----------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

acc_rf = accuracy_score(y_test, y_pred_rf)
print("\n--- Random Forest ---")
print("Accuracy:", acc_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf, labels=rf_model.classes_)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=rf_model.classes_)
disp_rf.plot(cmap='Greens')
plt.title("Random Forest Confusion Matrix")
plt.show()
print(classification_report(y_test, y_pred_rf))

# -----------------------------
# Support Vector Machine
# -----------------------------
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

acc_svm = accuracy_score(y_test, y_pred_svm)
print("\n--- Support Vector Machine ---")
print("Accuracy:", acc_svm)
cm_svm = confusion_matrix(y_test, y_pred_svm, labels=svm_model.classes_)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=svm_model.classes_)
disp_svm.plot(cmap='Oranges')
plt.title("SVM Confusion Matrix")
plt.show()
print(classification_report(y_test, y_pred_svm))

# -----------------------------
# Sample Prediction
# -----------------------------
sample = pd.DataFrame([[6.1, 3.0, 4.6, 1.4]], columns=X.columns)
sample_scaled = scaler.transform(sample)

print("\n--- Sample Prediction ---")
print("Input Sample:", sample.values[0])
print("Logistic Regression Prediction:", log_model.predict(sample_scaled)[0])
print("Random Forest Prediction:", rf_model.predict(sample_scaled)[0])
print("SVM Prediction:", svm_model.predict(sample_scaled)[0])

# -----------------------------
# Best Model Selection
# -----------------------------
print("\n--- Best Model ---")
accuracies = {
    "Logistic Regression": acc_log,
    "Random Forest": acc_rf,
    "SVM": acc_svm
}

best_model_name = max(accuracies, key=accuracies.get)
best_model_accuracy = accuracies[best_model_name]

print(f"Best Performing Model: {best_model_name}")
print(f"Accuracy: {best_model_accuracy:.4f}")
