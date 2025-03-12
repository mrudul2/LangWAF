import pandas as pd
import os

# Get project root dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Define file paths dynamically
input_csv_path = os.path.join(project_root, "data", "raw", "csic_database.csv")
output_csv_path = os.path.join(project_root, "data", "processed", "cleaned_csic2010.csv")

# Load dataset
if os.path.exists(input_csv_path):
    df = pd.read_csv(input_csv_path)
    print(f"✅ CSV loaded successfully from: {input_csv_path}")
else:
    print(f"❌ Error: CSV file not found at {input_csv_path}")
    exit()

# Rename columns
df.rename(columns={
    'Method': 'method',
    'content': 'payload',
    'lenght': 'length',  # Fix typo
    'classification': 'attack_type'
}, inplace=True)

# Select only relevant columns
df = df[['method', 'payload', 'length', 'attack_type']]

# Convert attack_type to binary (1 = Attack, 0 = Normal)
df['label'] = df['attack_type'].apply(lambda x: 1 if x != 'normal' else 0)

# Fill missing values
df.fillna("", inplace=True)

# Ensure processed directory exists
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

# Save cleaned dataset
df.to_csv(output_csv_path, index=False)
print(f" Cleaned dataset saved at: {output_csv_path}")
