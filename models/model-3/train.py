# import pandas as pd
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC

# def load_and_preprocess_data(dataset_path):
#     # Define column names
#     column_names = ["url", "detected_language", "label", "is_unsafe", "payload_length",
#                     "num_special_chars", "lang_match", "precision_m1", "recall_m1", "fpr_m1", "fnr_m1"]

#     # Check if file exists
#     if not os.path.exists(dataset_path):
#         raise FileNotFoundError(f"❌ Dataset not found at: {dataset_path}")

#     # Read CSV with headers
#     df = pd.read_csv(dataset_path, names=column_names, header=0, low_memory=False)

#     print("\n📌 Dataset Info Before Preprocessing:")
#     print(df.info())

#     # Handle missing values in 'label'
#     if df["label"].isnull().sum() > 0:
#         print(f"⚠️ Warning: Found {df['label'].isnull().sum()} missing labels. Replacing with 'unknown'.")
#         df["label"] = df["label"].fillna("unknown")  # Assign 'unknown' instead of 0 if unsure

#     df["label"] = df["label"].astype(str)  # Ensure label is string

#     # Convert numeric columns (fill NaN values with 0)
#     numeric_columns = ["detected_language", "is_unsafe", "payload_length", "num_special_chars",
#                        "lang_match", "precision_m1", "recall_m1", "fpr_m1", "fnr_m1"]

#     for col in numeric_columns:
#         df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

#     print("\n📌 Dataset Info After Preprocessing:")
#     print(df.info())

#     print(f"\n✅ Final dataset: {df.shape[0]} rows, {df.shape[1]} columns")
#     return df

# def train_models(df):
#     print("\n🚀 Starting Model Training...")

#     # Remove non-feature columns
#     X = df.drop(columns=["label", "url"], errors='ignore')  # Ignore if column is missing
#     y = df["label"]

#     if X.empty or y.empty:
#         raise ValueError("❌ Feature set or labels are empty after preprocessing.")

#     # Split dataset
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     print("✅ Dataset Split Done!")

#     # Train Decision Tree
#     print("⏳ Training Decision Tree...")
#     decision_tree = DecisionTreeClassifier(max_depth=5)
#     decision_tree.fit(X_train, y_train)
#     print("✅ Decision Tree Training Complete!")

#     # Train SVM
#     print("⏳ Training SVM (This might take time)...")
#     svm_model = SVC(kernel='rbf', probability=True)
#     svm_model.fit(X_train, y_train)
#     print("✅ SVM Training Complete!")

#     # Compute accuracy
#     dt_accuracy = decision_tree.score(X_test, y_test)
#     svm_accuracy = svm_model.score(X_test, y_test)

#     print("\n📊 Model Training Completed!")
#     print(f"🎯 Decision Tree Accuracy: {dt_accuracy:.4f}")
#     print(f"🎯 SVM Accuracy: {svm_accuracy:.4f}")

#     return decision_tree, svm_model, dt_accuracy, svm_accuracy

# if __name__ == "__main__":
#     dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/preprocessed_data-CISC_HTTPParams.csv"))

#     print("🚀 Loading Dataset...")
#     df = load_and_preprocess_data(dataset_path)

#     if df.empty:
#         print("❌ Error: No valid data after preprocessing. Check CSV formatting.")
#     else:
#         decision_tree, svm_model, dt_accuracy, svm_accuracy = train_models(df)
#         print(f"\n✅ Final Results:")
#         print(f"📊 Decision Tree Accuracy: {dt_accuracy:.2f}")
#         print(f"📊 SVM Accuracy: {svm_accuracy:.2f}")



# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report

# # Load Model-1 and Model-2 outputs
# def load_model_outputs(model1_path, model2_path):
#     model1_df = pd.read_csv(model1_path)
#     model2_df = pd.read_csv(model2_path)
#     return model1_df, model2_df

# # Feature Engineering and Dynamic Weighting
# def preprocess_and_merge(model1_df, model2_df):
#     print("Model 1 Columns:", model1_df.columns)
#     print("Model 2 Columns:", model2_df.columns)

#     df = model1_df.merge(model2_df, on='request_id', how='inner')  # Ensure we're merging correctly

#     print("Merged DataFrame Columns:", df.columns)
#     print("Merged DataFrame Shape:", df.shape)  # Check if it's empty

#     if df.empty:
#         raise ValueError("🚨 Error: Merged DataFrame is empty! Check if 'request_id' values match in both CSVs.")

#     classification_col = 'classification_x' if 'classification_x' in df.columns else 'classification_y'
#     df['safe'] = df[classification_col].apply(lambda x: 1 if x == 'Safe' else 0)

#     if 'confidence' not in df.columns:
#         print("⚠️ Warning: 'confidence' column missing! Assigning default value 0.5")
#         df['confidence'] = 0.5  # Assign a neutral confidence value

#     df['weight'] = df['confidence'].apply(lambda x: 0.3 if x < 0.5 else (0.7 if x < 0.8 else 1.0))
#     df['final_score'] = df['safe'] * df['weight']
#     df['final_label'] = df['final_score'].apply(lambda x: 'Bypassable' if x >= 0.7 else ('Needs Inspection' if x >= 0.3 else 'Blocked'))

#     return df[['request_id', 'safe', 'confidence', 'weight', 'final_score', 'final_label']]





# # Model Training
# def train_model(df):
#     X = df[['safe', 'confidence', 'weight']]
#     y = df['final_label']
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     dt_model = DecisionTreeClassifier()
#     dt_model.fit(X_train, y_train)
    
#     svm_model = SVC()
#     svm_model.fit(X_train, y_train)
    
#     return dt_model, svm_model, X_test, y_test

# # Evaluate the model
# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     print("Model Performance:")
#     print(classification_report(y_test, y_pred))
#     print("Accuracy:", accuracy_score(y_test, y_pred))

# if __name__ == "__main__":
#     model1_output_path = "../../data/model-1_results.csv"
#     model2_output_path = "../../data/model-2_data_with_languages-CISC_HTTPParams.csv"
    
#     model1_df, model2_df = load_model_outputs(model1_output_path, model2_output_path)
#     processed_df = preprocess_and_merge(model1_df, model2_df)
    
#     dt_model, svm_model, X_test, y_test = train_model(processed_df)
    
#     print("Decision Tree Evaluation:")
#     evaluate_model(dt_model, X_test, y_test)
    
#     print("\nSVM Evaluation:")
#     evaluate_model(svm_model, X_test, y_test)
    
#     # Save trained models
#     with open("decision_tree_model.pkl", "wb") as f:
#         pickle.dump(dt_model, f)
#     with open("svm_model.pkl", "wb") as f:
#         pickle.dump(svm_model, f)
    
#     print("Models saved successfully.")


import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(dataset_path):
    column_names = ["url", "detected_language", "label", "is_unsafe", "payload_length",
                    "num_special_chars", "lang_match", "precision_m1", "recall_m1", "fpr_m1", "fnr_m1", "confidence_m2"]

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ Dataset not found at: {dataset_path}")

    df = pd.read_csv(dataset_path, names=column_names, header=0, low_memory=False)

    print("\n📌 Dataset Info Before Preprocessing:")
    print(df.info())

    if df["label"].isnull().sum() > 0:
        print(f"⚠️ Warning: Found {df['label'].isnull().sum()} missing labels. Replacing with 'unknown'.")
        df["label"] = df["label"].fillna("unknown")

    df["label"] = df["label"].astype(str)

    numeric_columns = ["detected_language", "is_unsafe", "payload_length", "num_special_chars",
                       "lang_match", "precision_m1", "recall_m1", "fpr_m1", "fnr_m1", "confidence_m2"]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # ✅ Compute the weighted score and add it as a new feature
    df["weighted_score"] = df.apply(lambda row: compute_weighted_score(
        row["is_unsafe"], row["lang_match"], row["precision_m1"], row["recall_m1"], 
        row["fpr_m1"], row["fnr_m1"], row["confidence_m2"]), axis=1)

    print("\n📌 Dataset Info After Preprocessing:")
    print(df.info())

    print(f"\n✅ Final dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def compute_weighted_score(is_unsafe, lang_match, precision_m1, recall_m1, fpr_m1, fnr_m1, confidence_m2):
    W1 = ((precision_m1 + recall_m1) / 2) * (1 - fpr_m1)
    W2 = (1 - fnr_m1) * confidence_m2
    return (W1 + W2) / 2

def classify_request(final_score):
    if final_score > 0.8:
        return "Bypassable ✅"
    elif 0.5 <= final_score <= 0.8:
        return "Further Inspection ⚠️"
    else:
        return "Blocked ❌"

def train_models(df):
    print("\n🚀 Starting Model Training...")

    X = df.drop(columns=["label", "url"], errors='ignore')  
    y = df["label"]

    if X.empty or y.empty:
        raise ValueError("❌ Feature set or labels are empty after preprocessing.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("✅ Dataset Split Done!")

    # ✅ Feature Scaling (Important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("⏳ Training Decision Tree...")
    decision_tree = DecisionTreeClassifier(max_depth=5)
    decision_tree.fit(X_train, y_train)
    print("✅ Decision Tree Training Complete!")

    # ✅ Hyperparameter Optimization (Faster)
    print("⏳ Running Optimized Hyperparameter Search for SVM...")
    param_dist = {
        'C': np.logspace(-2, 2, 5),  # [0.01, 0.1, 1, 10, 100]
        'gamma': np.logspace(-3, 1, 5),  # [0.001, 0.01, 0.1, 1, 10]
        'kernel': ['linear', 'rbf']
    }

    svm_model = RandomizedSearchCV(SVC(probability=True, class_weight='balanced'), param_dist, n_iter=10, cv=3, n_jobs=-1, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    print(f"✅ Best SVM Parameters: {svm_model.best_params_}")

    dt_accuracy = decision_tree.score(X_test, y_test)
    svm_accuracy = svm_model.best_estimator_.score(X_test_scaled, y_test)

    print("\n📊 Model Training Completed!")
    print(f"🎯 Decision Tree Accuracy: {dt_accuracy:.4f}")
    print(f"🎯 SVM Accuracy: {svm_accuracy:.4f}")

    return decision_tree, svm_model.best_estimator_, dt_accuracy, svm_accuracy

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

        example_score = compute_weighted_score(is_unsafe=1, lang_match=0, precision_m1=0.85, recall_m1=0.90,
                                               fpr_m1=0.10, fnr_m1=0.05, confidence_m2=0.95)
        decision = classify_request(example_score)
        print(f"Example Decision: {decision}")

