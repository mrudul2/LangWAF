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

# Drop non-useful columns, keeping 'url'
df_model3 = df_model3.drop(columns=["request_id", "payload"])

# Save the preprocessed data
df_model3.to_csv(os.path.join(os.path.dirname(__file__), "../../data", "preprocessed_data-CISC_HTTPParams.csv"), index=False)

print("✅ Data preprocessing complete!")
