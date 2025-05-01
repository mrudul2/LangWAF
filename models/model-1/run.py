import pandas as pd
import joblib
import json
import os
import hashlib
import numpy as np
import time  # Import time module for performance measurement

data_dir = os.path.join(os.path.dirname(__file__), "../../data")
results_dir = os.path.join(os.path.dirname(__file__), "../../results")
os.makedirs(results_dir, exist_ok=True)

# Models to use
models = {
    "svm": "svm_model.pkl",
    "svm_prob": "svm_prob_model.pkl", 
    "logistic_regression": "logistic_regression_model.pkl",
    "random_forest": "random_forest_model.pkl"
}

# Load trained models
loaded_models = {}
for model_name, filename in models.items():
    model_path = os.path.join(os.path.dirname(__file__), f"../../models/model-1/{filename}")
    try:
        loaded_models[model_name] = joblib.load(model_path)
        print(f"Loaded {model_name} from {model_path}")
    except Exception as e:
        print(f"Warning: Could not load {model_name}: {e}")

# Load dataset
dataset_path = os.path.join(os.path.dirname(__file__), "../../data/CSIC_HTTPParams-model1.csv")
df = pd.read_csv(dataset_path)

feature_columns = ['payload_len', 'alpha', 'non_alpha', 'attack_feature']
X = df[feature_columns]

# Ensure required columns exist
if 'url' not in df.columns or 'payload' not in df.columns:
    raise ValueError("Dataset must contain 'url' and 'payload' columns.")

model_predictions = {}
model_probabilities = {}

# Make predictions with each model
for model_name, model in loaded_models.items():
    # Start timing
    start_time = time.perf_counter()
    
    if model_name == "svm":
        # Standard SVM only outputs class labels
        model_predictions[model_name] = model.predict(X)
        # No probabilities for standard SVM
        model_probabilities[model_name] = None
    else:
        # Models that can output probabilities
        model_predictions[model_name] = model.predict(X)
        # Get probability for class 1 (unsafe)
        probs = model.predict_proba(X)[:, 1] 
        model_probabilities[model_name] = probs
    
    # End timing and calculate execution time
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Model {model_name} execution time: {execution_time:.4f} seconds")

# Prepare output data for each model
for model_name in loaded_models.keys():
    output_data = []
    
    for index, prediction in enumerate(model_predictions[model_name]):
        request_data = df.iloc[index]  # Full row of data
        request_id = hashlib.md5(str(request_data.values).encode()).hexdigest()[:8]  # Unique ID
        
        # Get the probability score if available
        probability = None
        if model_probabilities[model_name] is not None:
            probability = float(model_probabilities[model_name][index])
        
        output_data.append({
            "request_id": request_id,
            "url": "" if pd.isnull(request_data["url"]) else request_data["url"], # URL from dataset
            "payload": "" if pd.isnull(request_data["payload"]) else request_data["payload"],  # Payload from dataset
            "classification": "unsafe" if prediction == 1 else "safe",
            "probability": probability,  # Add probability if available
            "label": "" if "label" not in df.columns or pd.isnull(request_data["label"]) else int(request_data["label"]) if isinstance(request_data["label"], (int, float)) else str(request_data["label"])
        })
    
    # Save JSON
    json_path = os.path.join(results_dir, f"classified_requests-{model_name}-CISC_HTTPParams.json")
    
    # save in csv format
    output_csv_path = os.path.join(data_dir, f"model-1_results-{model_name}-CISC_HTTPParams.csv")
    df_output = pd.DataFrame(output_data)
    df_output.to_csv(output_csv_path, index=False)
    print(f"Predictions for {model_name} saved to {output_csv_path}")
    
    with open(json_path, "w") as json_file:
        json.dump(output_data, json_file, indent=4)
    print(f"Predictions for {model_name} saved to {json_path}")

# For backward compatibility, save the SVM results to the original filename
original_json_path = os.path.join(results_dir, "classified_requests-CISC_HTTPParams.json")
with open(original_json_path, "w") as json_file:
    # Get the SVM results
    svm_output = [item for item in json.load(open(os.path.join(results_dir, "classified_requests-svm-CISC_HTTPParams.json")))]
    json.dump(svm_output, json_file, indent=4)
print(f"Original SVM predictions saved to {original_json_path}")

# Print a summary for each model
for model_name in loaded_models.keys():
    predictions = model_predictions[model_name]
    safe_count = sum(1 for p in predictions if p == 0)
    unsafe_count = sum(1 for p in predictions if p == 1)
    print(f"\nModel: {model_name}")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Safe: {safe_count} ({safe_count/len(predictions)*100:.2f}%)")
    print(f"  Unsafe: {unsafe_count} ({unsafe_count/len(predictions)*100:.2f}%)")
    
    if model_probabilities[model_name] is not None:
        probs = model_probabilities[model_name]
        avg_prob = np.mean(probs)
        print(f"  Average probability score: {avg_prob:.4f}")
        print(f"  Distribution of probability scores:")
        
        # Create buckets for visualization
        buckets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        counts = [sum(1 for p in probs if bucket-0.1 <= p < bucket) for bucket in buckets[1:]]
        
        for i, count in enumerate(counts):
            lower = buckets[i]
            upper = buckets[i+1]
            print(f"    {lower:.1f}-{upper:.1f}: {count} ({count/len(probs)*100:.2f}%)")
