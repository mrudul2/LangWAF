import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

dataset_path = os.path.join(os.path.dirname(__file__), "../../data/model-3_training_data.csv")
df_model3 = pd.read_csv(dataset_path)



# Encode 'detected_language'
label_enc = LabelEncoder()
df_model3.loc[:, "detected_language"] = df_model3["detected_language"].fillna("UNKNOWN")  # Fill NaNs
df_model3.loc[:, "detected_language"] = label_enc.fit_transform(df_model3["detected_language"])

# 1️⃣ Drop duplicate rows
df_model3 = df_model3.drop_duplicates()
print(f"✅ Removed duplicate rows. New shape: {df_model3.shape}")

# 2️⃣ Handle missing values in 'payload'
df_model3.loc[:, "payload"] = df_model3["payload"].fillna("UNKNOWN")  
print(f"✅ Handled missing values in 'payload'.")

# Drop non-useful columns (keep 'url' and 'payload' if needed)
df_model3 = df_model3.drop(columns=["request_id","url","payload"])  


# save the preprocessed data
df_model3.to_csv(os.path.join(os.path.dirname(__file__), "../../data", "preprocessed_data.csv"), index=False)

print("✅ Data preprocessing complete!")