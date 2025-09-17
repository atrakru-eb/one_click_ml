"""
Simple ML model training script for demonstration
"""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def train_model():
    """Train a simple classification model"""
    # Generate sample data (replace with your actual data loading)
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10, 
        n_redundant=10, 
        n_clusters_per_class=1, 
        random_state=42
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save the model
    os.makedirs('model/artifacts', exist_ok=True)
    joblib.dump(model, 'model/artifacts/model.pkl')
    print("Model saved to model/artifacts/model.pkl")
    
    return model, accuracy

if __name__ == "__main__":
    train_model()