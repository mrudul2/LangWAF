import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def load_and_preprocess_data(dataset_path):
    # Define column names
    column_names = ["url", "detected_language", "label", "is_unsafe", "payload_length",
                    "num_special_chars", "lang_match", "precision_m1", "recall_m1", "fpr_m1", "fnr_m1"]

    # Check if file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ Dataset not found at: {dataset_path}")

    # Read CSV with headers
    df = pd.read_csv(dataset_path, names=column_names, header=0, low_memory=False)

    print("\n📌 Dataset Info Before Preprocessing:")
    print(df.info())

    # Handle missing values in 'label'
    if df["label"].isnull().sum() > 0:
        print(f"⚠️ Warning: Found {df['label'].isnull().sum()} missing labels. Replacing with 'unknown'.")
        df["label"] = df["label"].fillna("unknown")  # Assign 'unknown' instead of 0 if unsure

    df["label"] = df["label"].astype(str)  # Ensure label is string

    # Convert numeric columns (fill NaN values with 0)
    numeric_columns = ["detected_language", "is_unsafe", "payload_length", "num_special_chars",
                       "lang_match", "precision_m1", "recall_m1", "fpr_m1", "fnr_m1"]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    print("\n📌 Dataset Info After Preprocessing:")
    print(df.info())

    print(f"\n✅ Final dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def train_models(df):
    print("\n🚀 Starting Model Training...")

    # Remove non-feature columns
    X = df.drop(columns=["label", "url"], errors='ignore')  # Ignore if column is missing
    y = df["label"]

    if X.empty or y.empty:
        raise ValueError("❌ Feature set or labels are empty after preprocessing.")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("✅ Dataset Split Done!")

    # Train Decision Tree
    print("⏳ Training Decision Tree...")
    decision_tree = DecisionTreeClassifier(max_depth=5)
    decision_tree.fit(X_train, y_train)
    print("✅ Decision Tree Training Complete!")

    # Train SVM
    print("⏳ Training SVM (This might take time)...")
    svm_model = SVC(kernel='rbf', probability=True)
    svm_model.fit(X_train, y_train)
    print("✅ SVM Training Complete!")

    # Compute accuracy
    dt_accuracy = decision_tree.score(X_test, y_test)
    svm_accuracy = svm_model.score(X_test, y_test)

    print("\n📊 Model Training Completed!")
    print(f"🎯 Decision Tree Accuracy: {dt_accuracy:.4f}")
    print(f"🎯 SVM Accuracy: {svm_accuracy:.4f}")

    return decision_tree, svm_model, dt_accuracy, svm_accuracy

if __name__ == "__main__":
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/preprocessed_data-CISC_HTTPParams.csv"))

    print("🚀 Loading Dataset...")
    df = load_and_preprocess_data(dataset_path)

    if df.empty:
        print("❌ Error: No valid data after preprocessing. Check CSV formatting.")
    else:
        decision_tree, svm_model, dt_accuracy, svm_accuracy = train_models(df)
        print(f"\n✅ Final Results:")
        print(f"📊 Decision Tree Accuracy: {dt_accuracy:.2f}")
        print(f"📊 SVM Accuracy: {svm_accuracy:.2f}")
