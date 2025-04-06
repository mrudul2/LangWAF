import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

dataset_path = os.path.join(os.path.dirname(__file__), "../../data/model-3_training_data.csv")
df_model3 = pd.read_csv(dataset_path)

# Encode 'detected_language'
label_enc = LabelEncoder()
df_model3["detected_language"] = df_model3["detected_language"].fillna("UNKNOWN")
df_model3["detected_language"] = label_enc.fit_transform(df_model3["detected_language"])

# 2️⃣ Handle missing values in 'payload'
df_model3["payload"] = df_model3["payload"].fillna("UNKNOWN")  
print(f"✅ Handled missing values in 'payload'.")

# Include probability as feature if available
if "probability" in df_model3.columns:
    print(f"✅ Found probability values from model-1.")
else:
    print(f"⚠️ No probability values found. Models will use binary classification only.")

# Keep request_id for tracking
id_column = df_model3["request_id"] if "request_id" in df_model3.columns else None

# Drop non-useful columns, but keep 'url' and probability if available
columns_to_drop = ["request_id", "payload"]
if "probability" not in df_model3.columns:
    columns_to_drop.append("probability")  # Only drop if not available

df_model3 = df_model3.drop(columns=columns_to_drop, errors='ignore')

# Save the preprocessed data with the original filename format
df_model3.to_csv(os.path.join(os.path.dirname(__file__), "../../data", "preprocessed_data-CISC_HTTPParams.csv"), index=False)

# Also save a version with the request_id for tracking results
if id_column is not None:
    df_with_id = df_model3.copy()
    df_with_id["request_id"] = id_column
    df_with_id.to_csv(os.path.join(os.path.dirname(__file__), "../../data", "preprocessed_data_with_id-CISC_HTTPParams.csv"), index=False)
    print(f"✅ Also saved preprocessed data with request_id for result tracking.")

print("✅ Data preprocessing complete!")

# Print feature columns for reference
print("\n📊 Available features for model training:")
for col in df_model3.columns:
    print(f"  - {col}")
