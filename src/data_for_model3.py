import pandas as pd
import json
import os

# Set paths
results_dir = os.path.join(os.path.dirname(__file__), "../results")
data_dir = os.path.join(os.path.dirname(__file__), "../data")

# Load Model-1 results (Safe/Unsafe classification)
classified_requests_path = os.path.join(results_dir, "classified_requests.json")
with open(classified_requests_path, "r") as file:
    model1_results = json.load(file)

# Load Model-2 results (Language Detection)
data_with_lang_path = os.path.join(results_dir, "data_with_languages.json")
with open(data_with_lang_path, "r") as file:
    model2_results = json.load(file)

# Load Model-1 performance metrics (Precision, Recall, FPR, FNR)
metrics_path = os.path.join(results_dir, "model1_results.json")
with open(metrics_path, "r") as file:
    model1_metrics = json.load(file)

precision_m1 = model1_metrics["precision"] / 100 # Convert to decimal
recall_m1 = model1_metrics["recall"] / 100 # Convert to decimal

# Calculate FPR and FNR
TN = model1_metrics["confusion_matrix"][0][0]
FP = model1_metrics["confusion_matrix"][0][1]
FN = model1_metrics["confusion_matrix"][1][0]
TP = model1_metrics["confusion_matrix"][1][1]
# Compute FPR & FNR
fpr_m1 = FP / (FP + TN) if (FP + TN) > 0 else 0  # Avoid division by zero
fnr_m1 = FN / (FN + TP) if (FN + TP) > 0 else 0  # Avoid division by zero



# Convert JSON data to DataFrame
df_model1 = pd.DataFrame(model1_results)
# df_model2 = pd.DataFrame(model2_results)

# load model-2 data
df_model2 = pd.read_csv(os.path.join(data_dir, "model-2_data_with_languages.csv"))
df_labels = pd.read_csv(os.path.join(data_dir, "request_id_label.csv"))
df_labels_grouped = df_labels.groupby("request_id").agg({"label": "max"}).reset_index() # Group by request_id and keep the max label

df_final = pd.merge(df_model2, df_labels_grouped, on="request_id", how="left")





# Convert classification labels to numeric values
df_final["is_unsafe"] = df_final["classification"].apply(lambda x: 1 if x == "unsafe" else 0)

# Extract numerical features
df_final["payload_length"] = df_final["payload"].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)
df_final["num_special_chars"] = df_final["payload"].apply(lambda x: sum(1 for c in str(x) if c in "<>;$()") if pd.notnull(x) else 0)
df_final["url_depth"] = df_final["url"].apply(lambda x: str(x).count('/') if pd.notnull(x) else 0)

# ==============================
# RULE-BASED EXPECTED LANGUAGE DETECTION
# ==============================

expected_languages = {
    "sql", "php", "javascript", "html"
}

df_final["lang_match"] = df_final["detected_language"].str.lower().isin(expected_languages).astype(int) # 1 if match, 0 if no match

# Add Model-1 performance metrics
df_final["precision_m1"] = precision_m1
df_final["recall_m1"] = recall_m1
df_final["fpr_m1"] = fpr_m1
df_final["fnr_m1"] = fnr_m1

# Drop unnecessary columns
df_final = df_final.drop(columns=["classification"])

# Save processed data for Model-3 training
output_csv_path = os.path.join(data_dir, "model-3_training_data.csv")
df_final.to_csv(output_csv_path, index=False)

print(f"✅ Model-3 training data saved to {output_csv_path}")
