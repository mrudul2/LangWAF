# To train model1 and save it to models/model-1/waf_model.pkl
# Run the following command:
# python models/model-1/train.py in the terminal

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# dataset_path = "C:/FYP-LangWAF/LangWAF/data/CSIC.csv" #having issues with relative path, so using absolute path -mj
dataset_path = os.path.join(os.path.dirname(__file__), "../../data/CSIC.csv")
df = pd.read_csv(dataset_path)

feature_columns = ['payload_len', 'alpha', 'non_alpha', 'attack_feature']
X = df[feature_columns]
y = df['label']

# (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM model
svm_model = SVC(kernel='linear', cache_size=7000)
svm_model.fit(X_train, y_train)
model_dir = os.path.join(os.path.dirname(__file__), "../../models/model-1")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "waf_model.pkl")
joblib.dump(svm_model, model_path)

print(f"Model saved to {model_path}")

# Eval
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))
