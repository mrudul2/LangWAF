import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, classification_report

def load_and_preprocess_data(dataset_path):
    column_names = ["url", "detected_language", "label", "is_unsafe", "payload_length",
                    "num_special_chars", "lang_match", "precision_m1", "recall_m1", "fpr_m1", "fnr_m1", "confidence_m2"]

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ Dataset not found at: {dataset_path}")

    df = pd.read_csv(dataset_path, names=column_names, header=0, low_memory=False)

    if df["label"].isnull().sum() > 0:
        df["label"] = df["label"].fillna("unknown")
    
    df["label"] = df["label"].astype(str)
    
    numeric_columns = ["detected_language", "is_unsafe", "payload_length", "num_special_chars",
                       "lang_match", "precision_m1", "recall_m1", "fpr_m1", "fnr_m1", "confidence_m2"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df["weighted_score"] = df.apply(lambda row: compute_weighted_score(
        row["is_unsafe"], row["lang_match"], row["precision_m1"], row["recall_m1"], 
        row["fpr_m1"], row["fnr_m1"], row["confidence_m2"]), axis=1)
    
    return df

def compute_weighted_score(is_unsafe, lang_match, precision_m1, recall_m1, fpr_m1, fnr_m1, confidence_m2):
    W1 = ((precision_m1 + recall_m1) / 2) * (1 - fpr_m1)
    W2 = (1 - fnr_m1) * confidence_m2
    return (W1 + W2) / 2

def train_models(df):
    X = df.drop(columns=["label", "url"], errors='ignore')  
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("⏳ Training Decision Tree...")
    decision_tree = DecisionTreeClassifier(max_depth=5)
    decision_tree.fit(X_train, y_train)
    dt_train_acc = decision_tree.score(X_train, y_train)
    dt_val_acc = decision_tree.score(X_test, y_test)
    dt_report = classification_report(y_test, decision_tree.predict(X_test))
    
    print("⏳ Running Optimized Hyperparameter Search for SVM...")
    param_dist = {
        'C': np.logspace(-2, 2, 5),
        'gamma': np.logspace(-3, 1, 5),
        'kernel': ['linear', 'rbf']
    }
    
    svm_model = RandomizedSearchCV(SVC(probability=True, class_weight='balanced'), param_dist, n_iter=10, cv=3, n_jobs=-1, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    best_svm = svm_model.best_estimator_
    svm_train_acc = best_svm.score(X_train_scaled, y_train)
    svm_val_acc = best_svm.score(X_test_scaled, y_test)
    svm_report = classification_report(y_test, best_svm.predict(X_test_scaled))
    
    print("✅ Best SVM Parameters:", svm_model.best_params_)
    print(f"🎯 Decision Tree Accuracy: Train={dt_train_acc:.4f}, Validation={dt_val_acc:.4f}")
    print(f"🎯 SVM Accuracy: Train={svm_train_acc:.4f}, Validation={svm_val_acc:.4f}")
    
    if dt_train_acc - dt_val_acc > 0.1:
        print("⚠️ Decision Tree might be overfitting!")
    else:
        print("✅ Decision Tree does not seem to be overfitting.")
    
    if svm_train_acc - svm_val_acc > 0.1:
        print("⚠️ SVM might be overfitting!")
    else:
        print("✅ SVM does not seem to be overfitting.")
    
    print("\n📊 Decision Tree Classification Report:\n", dt_report)
    print("\n📊 SVM Classification Report:\n", svm_report)
    
    return decision_tree, best_svm, dt_train_acc, dt_val_acc, svm_train_acc, svm_val_acc

if __name__ == "__main__":
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/preprocessed_data-CISC_HTTPParams.csv"))
    df = load_and_preprocess_data(dataset_path)
    if df.empty:
        print("❌ Error: No valid data after preprocessing.")
    else:
        decision_tree, svm_model, dt_train_acc, dt_val_acc, svm_train_acc, svm_val_acc = train_models(df)
        print(f"\n✅ Final Results:")
        print(f"📊 Decision Tree Accuracy: Train={dt_train_acc:.2f}, Validation={dt_val_acc:.2f}")
        print(f"📊 SVM Accuracy: Train={svm_train_acc:.2f}, Validation={svm_val_acc:.2f}")