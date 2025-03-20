import pandas as pd
import os

# Load dataset
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/model-3_training_data.csv"))
df = pd.read_csv(dataset_path)

# 1️⃣ Drop duplicate rows
df = df.drop_duplicates()
print(f"✅ Removed duplicate rows. New shape: {df.shape}")

# 2️⃣ Handle missing values in 'payload'
df['payload'] = df['payload'].fillna("UNKNOWN")  # You can change "UNKNOWN" to any placeholder
print(f"✅ Handled missing values in 'payload'.")

# Save cleaned dataset
cleaned_file_path =  os.path.join(os.path.dirname(__file__), "../../data/cleaned_model3_training_data.csv")
df.to_csv(cleaned_file_path, index=False)
print(f"✅ Cleaned dataset saved at: {cleaned_file_path}")