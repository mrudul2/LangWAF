import pandas as pd
import json
import os
import numpy as np

# Set paths
results_dir = os.path.join(os.path.dirname(__file__), "../results")
data_dir = os.path.join(os.path.dirname(__file__), "../data")

# Choose model to use for probabilities (options: svm_prob, logistic_regression, random_forest)
# Set to None to use the original SVM model without probabilities
PROBABILITY_MODEL = "random_forest"

# Load Model-1 results
if PROBABILITY_MODEL:
    # Try to load probability-based model results
    classified_requests_path = os.path.join(results_dir, f"classified_requests-{PROBABILITY_MODEL}-CISC_HTTPParams.json")
    try:
        with open(classified_requests_path, "r") as file:
            model1_results = json.load(file)
        print(f"Using {PROBABILITY_MODEL} model results for predictions and probabilities")
    except FileNotFoundError:
        # Fall back to standard SVM if probability model not found
        print(f"Could not find results from {PROBABILITY_MODEL} model, falling back to standard SVM")
        PROBABILITY_MODEL = None
else:
    # Use standard SVM results
    print("Using standard SVM model results (no probabilities available)")

# If no probability model found or explicitly set to None, use standard SVM
if not PROBABILITY_MODEL:
    classified_requests_path = os.path.join(results_dir, "classified_requests-CISC_HTTPParams.json")
    with open(classified_requests_path, "r") as file:
        model1_results = json.load(file)
    print("Using standard SVM model results (no probabilities available)")

# Load Model-2 results (Language Detection)
data_with_lang_path = os.path.join(results_dir, "data_with_languages-CISC_HTTPParams.json")
with open(data_with_lang_path, "r") as file:
    model2_results = json.load(file)

# Load Model-1 performance metrics
if PROBABILITY_MODEL:
    # Try to load metrics for the probability model first
    metrics_path = os.path.join(results_dir, f"model1_{PROBABILITY_MODEL}_results.json") 
    try:
        with open(metrics_path, "r") as file:
            model1_metrics = json.load(file)
        print(f"Using {PROBABILITY_MODEL} model metrics")
    except FileNotFoundError:
        # Fall back to standard SVM metrics
        PROBABILITY_MODEL = None

# If no probability model metrics found or explicitly set to None, use standard SVM metrics
if not PROBABILITY_MODEL:
    metrics_path = os.path.join(results_dir, "model1_results.json")
    with open(metrics_path, "r") as file:
        model1_metrics = json.load(file)
    print("Using standard SVM model metrics")

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

# Load model-2 data
df_model2 = pd.read_csv(os.path.join(data_dir, "model-2_data_with_languages-CISC_HTTPParams.csv"))
df_labels = pd.read_csv(os.path.join(data_dir, "request_id_label.csv"))

# Check if model-2 data has the confidence column
has_confidence = 'confidence' in df_model2.columns
if not has_confidence:
    print("⚠️ 'confidence' column not found in model-2 data. Using default confidence of 0.5.")

df_labels_grouped = df_labels.groupby("request_id").agg({"label": "max"}).reset_index() # Group by request_id and keep the max label

df_final = pd.merge(df_model2, df_labels_grouped, on="request_id", how="left", suffixes=[None, "_y"])

# Convert classification labels to numeric values
df_final["is_unsafe"] = df_final["classification"].apply(lambda x: 1 if x == "unsafe" else 0)

# Extract numerical features
df_final["payload_length"] = df_final["payload"].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)
df_final["num_special_chars"] = df_final["payload"].apply(lambda x: sum(1 for c in str(x) if c in "<>;$()") if pd.notnull(x) else 0)
df_final["url_depth"] = df_final["url"].apply(lambda x: str(x).count('/') if pd.notnull(x) else 0)

# Add prediction probability if available 
if PROBABILITY_MODEL and "probability" in df_model1.columns:
    # Merge probability from model1 results
    df_prob = df_model1[["request_id", "probability"]]
    df_final = pd.merge(df_final, df_prob, on="request_id", how="left")
    df_final["probability"] = df_final["probability"].fillna(0.5)  # Default to 0.5 for missing values
    print("✅ Added probability scores from model-1 to features")
else:
    # If probability not available, create a synthetic one based on classification
    # This is just for model compatibility, but won't provide real confidence info
    df_final["probability"] = df_final["is_unsafe"].apply(lambda x: 0.9 if x == 1 else 0.1)
    print("⚠️ No probability scores available. Using synthetic values based on classification (0.1/0.9)")

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

# Add confidence score from model-2 (if available)
if has_confidence:
    df_final["confidence_m2"] = df_final["confidence"].fillna(0.5)  # Use existing confidence or default to 0.5
else:
    # Create a default confidence value if the column doesn't exist
    df_final["confidence_m2"] = 0.5  # Default confidence
    print("✅ Created default confidence_m2 values of 0.5")

# Drop unnecessary columns
df_final = df_final.drop(columns=["classification"])

# Save processed data for Model-3 training
output_csv_path = os.path.join(data_dir, "model-3_training_data.csv")
df_final.to_csv(output_csv_path, index=False)

print(f"✅ Model-3 training data saved to {output_csv_path}")
print(f"✅ Data shape: {df_final.shape}")

# Print statistics about the probability distribution
print("\n📊 Probability Distribution:")
probs = df_final["probability"].values
print(f"  Min: {probs.min():.4f}")
print(f"  Max: {probs.max():.4f}")
print(f"  Mean: {probs.mean():.4f}")
print(f"  Median: {np.median(probs):.4f}")

# Create buckets for visualization
buckets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
counts = [sum(1 for p in probs if bucket-0.1 <= p < bucket) for bucket in buckets[1:]]

print("\n  Distribution by bucket:")
for i, count in enumerate(counts):
    lower = buckets[i]
    upper = buckets[i+1]
    print(f"    {lower:.1f}-{upper:.1f}: {count} ({count/len(probs)*100:.2f}%)")
    
# Display probability type
if PROBABILITY_MODEL:
    print(f"\n✅ Using actual probabilities from {PROBABILITY_MODEL} model")
else:
    print("\n⚠️ Using synthetic probabilities (0.1/0.9) based on SVM classifications")
