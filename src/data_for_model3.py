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

precision_m1 = model1_metrics["precision"]
recall_m1 = model1_metrics["recall"]
fpr_m1 = 1 - model1_metrics["precision"]  # Approximate FPR (1 - Precision)
fnr_m1 = 1 - model1_metrics["recall"]  # Approximate FNR (1 - Recall)

# Convert JSON data to DataFrame
df_model1 = pd.DataFrame(model1_results)
df_model2 = pd.DataFrame(model2_results)

# Merge Model-1 and Model-2 results on request_id
df = pd.merge(df_model1, df_model2[["request_id", "detected_language"]], on="request_id", how="left")

# Convert classification labels to numeric values
df["is_unsafe"] = df["classification"].apply(lambda x: 1 if x == "unsafe" else 0)

# Extract numerical features
df["payload_length"] = df["payload"].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)
df["num_special_chars"] = df["payload"].apply(lambda x: sum(1 for c in str(x) if c in "<>;$()") if pd.notnull(x) else 0)
df["url_depth"] = df["url"].apply(lambda x: str(x).count('/') if pd.notnull(x) else 0)

# ==============================
# ✅ RULE-BASED EXPECTED LANGUAGE DETECTION
# ==============================

expected_languages = {
    "/api/": "None",
    "/admin/": "PHP",
    "/db-query/": "SQL",
    "/static/": "None",
    "/scripts/": "JavaScript"
}

# Function to determine expected language based on URL pattern
def get_expected_lang(url):
    for pattern, lang in expected_languages.items():
        if pattern in url:
            return lang
    return "Unknown"  # Default if no match

# Apply rule-based expected language detection
df["expected_language"] = df["url"].apply(get_expected_lang)

# Compare detected language with expected language
df["lang_match"] = (df["detected_language"] == df["expected_language"]).astype(int)

# Add Model-1 performance metrics
df["precision_m1"] = precision_m1
df["recall_m1"] = recall_m1
df["fpr_m1"] = fpr_m1
df["fnr_m1"] = fnr_m1

# Drop unnecessary columns
df = df.drop(columns=["classification", "expected_language"])

# Save processed data for Model-3 training
output_csv_path = os.path.join(data_dir, "model3_training_data.csv")
df.to_csv(output_csv_path, index=False)

print(f"✅ Model-3 training data saved to {output_csv_path}")
