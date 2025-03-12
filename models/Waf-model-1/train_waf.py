import pandas as pd
import numpy as np
import joblib
import re
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Dynamically find project root and processed data path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

processed_data_path = os.path.join(project_root, "data", "processed", "cleaned_csic2010.csv")

# Load the preprocessed dataset
if os.path.exists(processed_data_path):
    df = pd.read_csv(processed_data_path)
    print(f" CSV loaded successfully from: {processed_data_path}")
else:
    print(f"Error: CSV file not found at {processed_data_path}")
    exit()

# Convert 'length' column to numerical format
def extract_length(value):
    if pd.isna(value):
        return 0
    match = re.search(r'\d+', str(value))
    return float(match.group()) if match else 0

df['length'] = df['length'].apply(extract_length)

# Convert 'payload' to numerical features using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)  
X_text = vectorizer.fit_transform(df['payload'].astype(str))

# Encode 'method' column (GET, POST, etc.)
method_encoder = LabelEncoder()
X_method = method_encoder.fit_transform(df['method'])

# Convert 'length' to NumPy array
X_length = df['length'].values.reshape(-1, 1)

# Combine all features
X = np.hstack((X_text.toarray(), X_method.reshape(-1, 1), X_length))

# Encode 'attack_type' labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['attack_type'])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Ensure models directory exists
models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

# Save the trained model and encoders
joblib.dump(model, os.path.join(models_dir, "waf_model.pkl"))
joblib.dump(vectorizer, os.path.join(models_dir, "vectorizer.pkl"))
joblib.dump(method_encoder, os.path.join(models_dir, "method_encoder.pkl"))
joblib.dump(label_encoder, os.path.join(models_dir, "label_encoder.pkl"))

# Debugging Output
print(f" Model training completed. Saved to '{models_dir}'.")
print(f" Feature Shape: {X.shape}, Train: {X_train.shape}, Test: {X_test.shape}")
print(f" Classes: {label_encoder.classes_}")
