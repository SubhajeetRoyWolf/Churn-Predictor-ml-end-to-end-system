import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Define the base directory (the folder where this script lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data():
    # Looks for the CSV in the same folder as the script
    csv_path = os.path.join(BASE_DIR, "cleaned_olist_master.csv")
    df = pd.read_csv(csv_path)
    return df

def preprocess(df):
    # Logic to create the target variable 'low_review'
    df['low_review'] = df['review_score'].apply(lambda x: 1 if x <= 2 else 0)
    
    X = df[["price", "freight_value", "delivery_time"]]
    y = df['low_review']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def save_model(model):
    # Saves to the 'model' subfolder relative to the project root
    model_path = os.path.join(BASE_DIR, "model", "olist_pipeline.pkl")
    
    # Create the directory if it doesn't exist to prevent errors
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    joblib.dump(model, model_path)

def run_pipeline():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    model = train_model(X_train, y_train)
    save_model(model)

if __name__ == "__main__":
    run_pipeline()