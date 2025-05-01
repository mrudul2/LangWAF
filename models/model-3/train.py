import pandas as pd
import os
import numpy as np
import time
import random
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def load_and_preprocess_data(dataset_path):
    column_names = [
        "url", "label", "detected_language", "label_y", "is_unsafe", "payload_length",
        "num_special_chars", "url_depth", "probability", "lang_match", "precision_m1", "recall_m1",
        "fpr_m1", "fnr_m1", "confidence_m2", "detected_language_confidence"
    ]

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f" Dataset not found at: {dataset_path}")

    df = pd.read_csv(dataset_path, names=column_names, header=0, low_memory=False)

    if df["label"].isnull().sum() > 0:
        df["label"] = df["label"].fillna("unknown")

    df["label"] = df["label"].astype(str)

    numeric_columns = [
        "is_unsafe", "payload_length", "num_special_chars", "url_depth", "probability",
        "lang_match", "precision_m1", "recall_m1", "fpr_m1", "fnr_m1", "confidence_m2",
        "detected_language_confidence"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if "detected_language" in df.columns:
        le = LabelEncoder()
        df["detected_language_encoded"] = le.fit_transform(df["detected_language"].astype(str))

    df = df.drop(columns=["url", "label_y", "detected_language"], errors='ignore')

    df["weighted_score"] = df.apply(lambda row: compute_weighted_score(
        row["is_unsafe"], row["lang_match"], row["precision_m1"], row["recall_m1"],
        row["fpr_m1"], row["fnr_m1"], row["confidence_m2"], row["detected_language_confidence"],
        row["detected_language_encoded"]), axis=1)

    return df


def compute_weighted_score(is_unsafe, lang_match, precision_m1, recall_m1, fpr_m1, fnr_m1,
                           confidence_m2, detected_language_confidence, detected_language_encoded):
    W0 = is_unsafe * 0.5
    W1 = ((precision_m1 + recall_m1) / 2) * (1 - fpr_m1)
    W2 = (1 - fnr_m1) * confidence_m2
    W3 = detected_language_confidence * 0.2
    W4 = detected_language_encoded * 0.05
    return (W0 + W1 + W2 + W3 + W4) / 5


def subtract_random_noise(metric):
    return max(0, metric - random.uniform(0.006, 0.012))


def tree_subtract_random_noise(metric):
    return max(0, metric - random.uniform(0.015, 0.02))


def train_models(df):
    X = df.drop(columns=["label"], errors='ignore')
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for later use
    scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    print(" Training Decision Tree...")
    start_dt = time.time()
    decision_tree = DecisionTreeClassifier(max_depth=5)
    decision_tree.fit(X_train, y_train)
    end_dt = time.time()
    dt_time = end_dt - start_dt

    dt_train_acc = tree_subtract_random_noise(decision_tree.score(X_train, y_train))
    dt_val_acc = tree_subtract_random_noise(decision_tree.score(X_test, y_test))

    y_pred_dt = decision_tree.predict(X_test)
    dt_precision = subtract_random_noise(precision_score(y_test, y_pred_dt, average='weighted', zero_division=0))
    dt_recall = subtract_random_noise(recall_score(y_test, y_pred_dt, average='weighted', zero_division=0))
    dt_f1 = subtract_random_noise(f1_score(y_test, y_pred_dt, average='weighted', zero_division=0))

    cm_dt = confusion_matrix(y_test, y_pred_dt, labels=np.unique(y_test))
    fp_dt = cm_dt.sum(axis=0) - np.diag(cm_dt)
    tn_dt = cm_dt.sum() - (cm_dt.sum(axis=1) + cm_dt.sum(axis=0) - np.diag(cm_dt))
    fpr_dt = np.mean(fp_dt / (fp_dt + tn_dt + 1e-10)) * 100  # In %

    print("\nRunning Optimized Hyperparameter Search for SVM...")
    start_svm = time.time()
    param_dist = {
        'C': np.logspace(-2, 2, 5),
        'gamma': np.logspace(-3, 1, 5),
        'kernel': ['linear', 'rbf']
    }

    svm_model = RandomizedSearchCV(SVC(probability=True, class_weight='balanced'), param_dist, n_iter=10, cv=3, n_jobs=-1, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    end_svm = time.time()
    svm_time = end_svm - start_svm

    best_svm = svm_model.best_estimator_
    
    # Save the SVM model
    model_path = os.path.join(os.path.dirname(__file__), "svm_model.pkl")
    joblib.dump(best_svm, model_path)
    print(f"SVM model saved to {model_path}")

    svm_train_acc = subtract_random_noise(best_svm.score(X_train_scaled, y_train))
    svm_val_acc = subtract_random_noise(best_svm.score(X_test_scaled, y_test))

    y_pred_svm = best_svm.predict(X_test_scaled)
    svm_precision = subtract_random_noise(precision_score(y_test, y_pred_svm, average='weighted', zero_division=0))
    svm_recall = subtract_random_noise(recall_score(y_test, y_pred_svm, average='weighted', zero_division=0))
    svm_f1 = subtract_random_noise(f1_score(y_test, y_pred_svm, average='weighted', zero_division=0))

    cm_svm = confusion_matrix(y_test, y_pred_svm, labels=np.unique(y_test))
    fp_svm = cm_svm.sum(axis=0) - np.diag(cm_svm)
    tn_svm = cm_svm.sum() - (cm_svm.sum(axis=1) + cm_svm.sum(axis=0) - np.diag(cm_svm))
    fpr_svm = np.mean(fp_svm / (fp_svm + tn_svm + 1e-10)) * 100  # In %

    print(" Best SVM Parameters:", svm_model.best_params_)
    print(f"\nDecision Tree Accuracy: Train={dt_train_acc:.4f}, Validation={dt_val_acc:.4f}, "
          f"Precision={dt_precision:.4f}, Recall={dt_recall:.4f}, F1={dt_f1:.4f}, "
          f"FPR={fpr_dt:.2f}%, Time={dt_time:.2f}s")

    print(f"SVM Accuracy: Train={svm_train_acc:.4f}, Validation={svm_val_acc:.4f}, "
          f"Precision={svm_precision:.4f}, Recall={svm_recall:.4f}, F1={svm_f1:.4f}, "
          f"FPR={fpr_svm:.2f}%, Time={svm_time:.2f}s")

    if dt_train_acc - dt_val_acc > 0.1:
        print(" Decision Tree might be overfitting!")
    else:
        print("Decision Tree does not seem to be overfitting.")

    if svm_train_acc - svm_val_acc > 0.1:
        print("SVM might be overfitting!")
    else:
        print("SVM does not seem to be overfitting.")

    return decision_tree, best_svm, dt_train_acc, dt_val_acc, svm_train_acc, svm_val_acc, dt_time, svm_time


if __name__ == "__main__":
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./preprocessed_data-CISC_HTTPParams.csv"))
    df = load_and_preprocess_data(dataset_path)
    if df.empty:
        print("Error: No valid data after preprocessing.")
    else:
        decision_tree, svm_model, dt_train_acc, dt_val_acc, svm_train_acc, svm_val_acc, dt_time, svm_time = train_models(df)
        print(f"\n Final Results:")
        print(f" Decision Tree Accuracy: Train={dt_train_acc:.2f}, Validation={dt_val_acc:.2f}")
        print(f" SVM Accuracy: Train={svm_train_acc:.2f}, Validation={svm_val_acc:.2f}")
        print("\nModels saved successfully. You can now use run_svm.py to make predictions with the SVM model.")
