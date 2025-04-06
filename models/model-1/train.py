from math import log
import pandas as pd
import os
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import logging
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import numpy as np

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

# Define models to train
models = {
    "svm": LinearSVC(dual=False, max_iter=5000),
    "svm_prob": SVC(probability=True, max_iter=5000),
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
probabilities = {}

# Train models
for model_name, model in models.items():
    print(f"Training {model_name}...")
    logger.info(f"Training {model_name}...")
    model.fit(X_train, y_train)
    
    # Save trained model
    model_dir = os.path.join(os.path.dirname(__file__), "../../models/model-1")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
    joblib.dump(model, model_path)
    print(f"{model_name} model saved to {model_path}")
    logger.info(f"{model_name} model saved to {model_path}")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Save probability predictions for probability-capable models
    if model_name != "svm":
        probabilities[model_name] = model.predict_proba(X_test)[:, 1]
    
    # Compute Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calculate ROC AUC for probability models
    roc_auc = None
    if model_name != "svm":
        roc_auc = roc_auc_score(y_test, probabilities[model_name])
    
    # Extract classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Store results
    results[model_name] = {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "roc_auc": round(roc_auc * 100, 2) if roc_auc is not None else None,
        "classification_report": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1-score": report[label]["f1-score"],
                "support": report[label]["support"]
            } for label in report if label not in ["accuracy", "macro avg", "weighted avg"]
        },
        "macro avg": {
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1-score": report["macro avg"]["f1-score"],
            "support": report["macro avg"]["support"]
        },
        "weighted avg": {
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1-score": report["weighted avg"]["f1-score"],
            "support": report["weighted avg"]["support"]
        },
        "confusion_matrix": conf_matrix.tolist()
    }

# Define the results directory
results_dir = os.path.join(os.path.dirname(__file__), "../../results")
os.makedirs(results_dir, exist_ok=True)

# Save each model's results
for model_name, result in results.items():
    json_results_path = os.path.join(results_dir, f"model1_{model_name}_results.json")
    
    with open(json_results_path, "w") as json_file:
        json.dump(result, json_file, indent=4)
    
    print(f"{model_name} results saved to {json_results_path}")
    logger.info(f"{model_name} results saved to {json_results_path}")
    
    # Plot and Save Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(result["confusion_matrix"], annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Safe', 'Unsafe'], yticklabels=['Safe', 'Unsafe'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(os.path.join(results_dir, f"confusion_matrix_{model_name}.png"))
    plt.close()

# For backward compatibility, save the SVM results to the original filename
with open(os.path.join(results_dir, "model1_results.json"), "w") as json_file:
    json.dump(results["svm"], json_file, indent=4)

# Compare models
print("\n=== Model Comparison ===")
for model_name, result in results.items():
    roc_info = f", ROC AUC: {result['roc_auc']}%" if result["roc_auc"] is not None else ""
    print(f"{model_name}: Accuracy: {result['accuracy']}%, Precision: {result['precision']}%, Recall: {result['recall']}%, F1: {result['f1_score']}%{roc_info}")

# Create comparison bar chart
plt.figure(figsize=(12, 6))
model_names = list(results.keys())
metrics = ['accuracy', 'precision', 'recall', 'f1_score']
metric_values = {metric: [results[model][metric] for model in model_names] for metric in metrics}

x = np.arange(len(model_names))
width = 0.2
multiplier = 0

for metric, values in metric_values.items():
    offset = width * multiplier
    rects = plt.bar(x + offset, values, width, label=metric.capitalize())
    multiplier += 1

plt.xlabel('Models')
plt.ylabel('Score (%)')
plt.title('Model Performance Comparison')
plt.xticks(x + width, model_names)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "model_comparison.png"))
plt.close()

print(f"\nModel comparison chart saved to {os.path.join(results_dir, 'model_comparison.png')}")
logger.info(f"Model comparison chart saved to {os.path.join(results_dir, 'model_comparison.png')}")
