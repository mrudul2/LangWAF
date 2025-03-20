import pandas as pd
import joblib
import json
import os
import hashlib

data_dir = os.path.join(os.path.dirname(__file__), "../../data")

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "../../models/model-1/waf_model.pkl")
svm_model = joblib.load(model_path)

# Load dataset
dataset_path = os.path.join(os.path.dirname(__file__), "../../data/CSIC.csv")
df = pd.read_csv(dataset_path)

feature_columns = ['payload_len', 'alpha', 'non_alpha', 'attack_feature']
X = df[feature_columns]

# Ensure required columns exist
if 'url' not in df.columns or 'payload' not in df.columns:
    raise ValueError("Dataset must contain 'url' and 'payload' columns.")

# Make predictions
predictions = svm_model.predict(X)

# JSON output
output_data = []
for index, prediction in enumerate(predictions):
    request_data = df.iloc[index]  # Full row of data
    request_id = hashlib.md5(str(request_data.values).encode()).hexdigest()[:8]  # Unique ID

    output_data.append({
        "request_id": request_id,
        "url": request_data["url"], # URL from dataset
        "payload": "" if pd.isnull(request_data["payload"]) else request_data["payload"],  # Payload from dataset
        "classification": "unsafe" if prediction == 1 else "safe"
    })

# Save JSON
results_dir = os.path.join(os.path.dirname(__file__), "../../results")
os.makedirs(results_dir, exist_ok=True)
json_path = os.path.join(results_dir, "classified_requests.json")

# save in csv format
output_csv_path = os.path.join(data_dir, "model-1_results.csv")
df = pd.DataFrame(output_data)
df.to_csv(output_csv_path, index=False)
print(f"Predictions saved to {output_csv_path}!!!!!!!")


with open(json_path, "w") as json_file:
    json.dump(output_data, json_file, indent=4)

print(f"Predictions saved to {json_path}")
