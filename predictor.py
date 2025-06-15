# predictor.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print("Dataset loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{filepath}'")
        return None

def feature_engineering(data):
    print("Performing feature engineering...")
    # Using k-mers (3-grams of characters) to convert sequences into numerical features
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
    X_features = vectorizer.fit_transform(data['sequence'])
    y_labels = data['label']
    
    # Save the vectorizer to use it later for new predictions
    joblib.dump(vectorizer, 'peptide_vectorizer.joblib')
    print("Feature engineering complete. Vectorizer saved.")
    return X_features, y_labels

def train_and_evaluate_model(X, y):
    print("\nSplitting data and training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")

    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Non-Antigenic (0)', 'Antigenic (1)'])
    
    print(f"\n--- Model Evaluation Results ---")
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    return model

def save_model(model, filename):
    print(f"\nSaving trained model to '{filename}'...")
    joblib.dump(model, filename)
    print("Model saved successfully.")

def main():
    print("--- Antigenic Peptide Predictor ---")
    dataset_path = "peptides_dataset.csv"
    
    data = load_data(dataset_path)
    if data is None:
        return

    X_features, y_labels = feature_engineering(data)
    
    trained_model = train_and_evaluate_model(X_features, y_labels)
    
    save_model(trained_model, "antigen_predictor_model.joblib")
    
    print("\n--- Project Execution Finished ---")

if __name__ == "__main__":
    main()