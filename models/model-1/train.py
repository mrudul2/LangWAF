from math import log
import pandas as pd
import os
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import logging
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"model-1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s',
  filename=log_file,
)
logger = logging.getLogger(__name__)

# Load dataset
# dataset_path = os.path.join(os.path.dirname(__file__), "../../data/CSIC.csv")
dataset_path = os.path.join(os.path.dirname(__file__), "../../data/CSIC_HTTPParams-model1.csv")
df = pd.read_csv(dataset_path)
logger.info(f"Dataset loaded from {dataset_path}")

# Define features and target variable
feature_columns = ['payload_len', 'alpha', 'non_alpha', 'attack_feature']
X = df[feature_columns]
y = df['label']

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logger.info("Dataset split into training and testing sets")
# Train SVM model
svm_model = LinearSVC(dual=False, max_iter=5000)
svm_model.fit(X_train, y_train)

# Save trained model
model_dir = os.path.join(os.path.dirname(__file__), "../../models/model-1")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "waf_model.pkl")
joblib.dump(svm_model, model_path)
print(f"Model saved to {model_path}")
logger.info(f"Model saved to {model_path}")

# Predictions
y_pred = svm_model.predict(X_test)

# Compute Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Save results in JSON
results = {
    "accuracy": round(accuracy * 100, 2),
    "precision": round(precision * 100, 2),
    "recall": round(recall * 100, 2),
    "f1_score": round(f1 * 100, 2),
    "classification_report": classification_report(y_test, y_pred, output_dict=True),
    "confusion_matrix": conf_matrix.tolist()
} 

logger.info(f"Results compute")

# Define results directory and file path
results_dir = os.path.join(os.path.dirname(__file__), "../../results")
os.makedirs(results_dir, exist_ok=True)
json_results_path = os.path.join(results_dir, "model1_results.json")

# Save JSON file
with open(json_results_path, "w") as json_file:
    json.dump(results, json_file, indent=4)

print(f"Results saved to {json_results_path}")

# Plot and Save Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Safe', 'Unsafe'], yticklabels=['Safe', 'Unsafe'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.show()
