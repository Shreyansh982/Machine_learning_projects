import sys
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from preprocess import load_and_process_data

# Define Paths
TRAIN_PATH = "data/train.tsv"
VALID_PATH = "data/valid.tsv"
TEST_PATH = "data/test.tsv"

def train_model():
    print("1. Loading Data...")
    cols = [
        'id', 'label', 'statement', 'subject', 'speaker', 'job_title', 
        'state_info', 'party_affiliation', 'barely_true_counts', 
        'false_counts', 'half_true_counts', 'mostly_true_counts', 
        'pants_on_fire_counts', 'context'
    ]
    def get_data(path):
        df = pd.read_csv(path, sep='\t', header=None, names=cols)
        df = load_and_process_data(path) 
        num_cols = ['barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts']
        for c in num_cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        return df

    train_df = get_data(TRAIN_PATH)
    valid_df = get_data(VALID_PATH)
    test_df = get_data(TEST_PATH)

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2)), 'combined_text'),
            ('num', StandardScaler(), ['barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts'])
        ]
    )

    mlflow.set_experiment("LIAR_Fake_News_RF")

    with mlflow.start_run():
        print("\n2. Training Random Forest...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1, class_weight='balanced'))
        ])
        pipeline.fit(train_df, train_df['target'])
        val_preds = pipeline.predict(valid_df)
        val_acc = accuracy_score(valid_df['target'], val_preds)
        val_f1 = f1_score(valid_df['target'], val_preds)
        
        print(f"      Validation Accuracy: {val_acc:.2%}")
        print(f"      Validation F1 Score: {val_f1:.4f}")

        test_preds = pipeline.predict(test_df)
        test_acc = accuracy_score(test_df['target'], test_preds)
        print(f"      Final Test Accuracy: {test_acc:.2%}")

        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.sklearn.log_model(pipeline, "model")
        print("\nModel saved.")

if __name__ == "__main__":
    train_model()