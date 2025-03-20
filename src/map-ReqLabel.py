import pandas as pd
import os
import hashlib


dataset_path = os.path.join(os.path.dirname(__file__), "../data/CSIC.csv")
data_dir = os.path.join(os.path.dirname(__file__), "../data")

df = pd.read_csv(dataset_path)
output_data = []

# loop through the dataset and save output as a csv of request_id and label
for index, row in df.iterrows():
    request_data = df.iloc[index]  # Full row of data
    request_id = hashlib.md5(str(request_data.values).encode()).hexdigest()[:8]  # Unique ID

    output_data.append({
        "request_id": request_id,
        "label": request_data["label"]
    })

output_csv_path = os.path.join(data_dir, "request_id_label.csv")
df = pd.DataFrame(output_data)
df.to_csv(output_csv_path, index=False)
print(f"Request ID and label saved to {output_csv_path}")

