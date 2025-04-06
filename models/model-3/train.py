import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, classification_report, roc_auc_score, accuracy_score
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ Dataset not found at: {dataset_path}")

    df = pd.read_csv(dataset_path, low_memory=False)
    
    if df["label"].isnull().sum() > 0:
        df["label"] = df["label"].fillna("unknown")
    
    df["label"] = df["label"].astype(str)
    
    # Handle numeric columns
    numeric_columns = [col for col in df.columns if col not in ["url", "label"]]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Compute weighted score - use probability if available
    if "probability" in df.columns:
        print("✅ Using probability scores from model-1 in weighted scoring")
        df["weighted_score"] = df.apply(lambda row: compute_weighted_score_with_prob(
            row["is_unsafe"], row["lang_match"], row["precision_m1"], row["recall_m1"], 
            row["fpr_m1"], row["fnr_m1"], row["confidence_m2"], row["probability"]), axis=1)
    else:
        print("⚠️ No probability scores available, using standard weighted scoring")
        df["weighted_score"] = df.apply(lambda row: compute_weighted_score(
            row["is_unsafe"], row["lang_match"], row["precision_m1"], row["recall_m1"], 
            row["fpr_m1"], row["fnr_m1"], row["confidence_m2"]), axis=1)
    
    return df

def compute_weighted_score(is_unsafe, lang_match, precision_m1, recall_m1, fpr_m1, fnr_m1, confidence_m2):
    W1 = ((precision_m1 + recall_m1) / 2) * (1 - fpr_m1)
    W2 = (1 - fnr_m1) * confidence_m2
    return (W1 + W2) / 2

def compute_weighted_score_with_prob(is_unsafe, lang_match, precision_m1, recall_m1, fpr_m1, fnr_m1, confidence_m2, probability):
    # Traditional weighted score
    W1 = ((precision_m1 + recall_m1) / 2) * (1 - fpr_m1)
    W2 = (1 - fnr_m1) * confidence_m2
    traditional = (W1 + W2) / 2
    
    # Weight based on probability (how confident model-1 is in its prediction)
    # Convert prediction confidence to a [0-1] scale where values closer to 1 
    # indicate higher confidence (either close to 0 or close to 1)
    confidence = abs(probability - 0.5) * 2
    
    # Combine traditional weighted score with probability confidence
    return traditional * 0.7 + confidence * 0.3

def train_models(df):
    X = df.drop(columns=["label", "url"], errors='ignore')  
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for use during prediction
    joblib.dump(scaler, os.path.join(os.path.dirname(__file__), "scaler.pkl"))
    
    print("⏳ Training Decision Tree...")
    decision_tree = DecisionTreeClassifier(max_depth=5)
    decision_tree.fit(X_train, y_train)
    dt_train_acc = decision_tree.score(X_train, y_train)
    dt_val_acc = decision_tree.score(X_test, y_test)
    dt_report = classification_report(y_test, decision_tree.predict(X_test))
    
    print("⏳ Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_train_acc = rf_model.score(X_train, y_train)
    rf_val_acc = rf_model.score(X_test, y_test)
    rf_report = classification_report(y_test, rf_model.predict(X_test))
    
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
    
    print("⏳ Training Gradient Boosting Classifier...")
    gbc_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    gbc_model.fit(X_train, y_train)
    gbc_train_acc = gbc_model.score(X_train, y_train)
    gbc_val_acc = gbc_model.score(X_test, y_test)
    gbc_report = classification_report(y_test, gbc_model.predict(X_test))
    
    # Calculate ROC AUC for each model
    dt_roc_auc = roc_auc_score(y_test, decision_tree.predict_proba(X_test)[:, 1]) if len(np.unique(y_test)) == 2 else None
    rf_roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]) if len(np.unique(y_test)) == 2 else None
    svm_roc_auc = roc_auc_score(y_test, best_svm.predict_proba(X_test_scaled)[:, 1]) if len(np.unique(y_test)) == 2 else None
    gbc_roc_auc = roc_auc_score(y_test, gbc_model.predict_proba(X_test)[:, 1]) if len(np.unique(y_test)) == 2 else None
    
    print("✅ Best SVM Parameters:", svm_model.best_params_)
    print(f"🎯 Decision Tree Accuracy: Train={dt_train_acc:.4f}, Validation={dt_val_acc:.4f}")
    print(f"🎯 Random Forest Accuracy: Train={rf_train_acc:.4f}, Validation={rf_val_acc:.4f}")
    print(f"🎯 SVM Accuracy: Train={svm_train_acc:.4f}, Validation={svm_val_acc:.4f}")
    print(f"🎯 GBC Accuracy: Train={gbc_train_acc:.4f}, Validation={gbc_val_acc:.4f}")
    
    # Save all models
    joblib.dump(decision_tree, os.path.join(os.path.dirname(__file__), "decision_tree_model.pkl"))
    joblib.dump(rf_model, os.path.join(os.path.dirname(__file__), "random_forest_model.pkl"))
    joblib.dump(best_svm, os.path.join(os.path.dirname(__file__), "svm_model.pkl"))
    joblib.dump(gbc_model, os.path.join(os.path.dirname(__file__), "gbc_model.pkl"))
    
    # Save feature names for later use
    with open(os.path.join(os.path.dirname(__file__), "feature_names.json"), 'w') as f:
        json.dump(list(X.columns), f)
    
    # Evaluate models and check for overfitting
    models_eval = {
        "Decision Tree": {
            "train_acc": dt_train_acc,
            "val_acc": dt_val_acc,
            "report": dt_report,
            "roc_auc": dt_roc_auc
        },
        "Random Forest": {
            "train_acc": rf_train_acc,
            "val_acc": rf_val_acc,
            "report": rf_report,
            "roc_auc": rf_roc_auc
        },
        "SVM": {
            "train_acc": svm_train_acc,
            "val_acc": svm_val_acc,
            "report": svm_report,
            "roc_auc": svm_roc_auc
        },
        "Gradient Boosting": {
            "train_acc": gbc_train_acc,
            "val_acc": gbc_val_acc,
            "report": gbc_report,
            "roc_auc": gbc_roc_auc
        }
    }
    
    for model_name, eval_data in models_eval.items():
        if eval_data["train_acc"] - eval_data["val_acc"] > 0.1:
            print(f"⚠️ {model_name} might be overfitting!")
        else:
            print(f"✅ {model_name} does not seem to be overfitting.")
        print(f"📊 {model_name} Classification Report:\n{eval_data['report']}")
        if eval_data["roc_auc"]:
            print(f"🎯 {model_name} ROC AUC: {eval_data['roc_auc']:.4f}")
    
    # Create comparison bar chart
    plt.figure(figsize=(12, 6))
    model_names = list(models_eval.keys())
    metrics = {
        'Training': [models_eval[m]["train_acc"] for m in model_names],
        'Validation': [models_eval[m]["val_acc"] for m in model_names],
        'ROC AUC': [models_eval[m]["roc_auc"] if models_eval[m]["roc_auc"] else 0 for m in model_names]
    }

    x = np.arange(len(model_names))
    width = 0.25
    multiplier = 0

    fig, ax = plt.subplots(figsize=(10, 6))

    for metric, values in metrics.items():
        offset = width * multiplier
        ax.bar(x + offset, values, width, label=metric)
        multiplier += 1

    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names)
    ax.legend(loc='upper left')
    plt.tight_layout()

    # Save the comparison chart
    results_dir = os.path.join(os.path.dirname(__file__), "../../results")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "model3_comparison.png"))
    
    # For backward compatibility, return the original models
    return decision_tree, best_svm, dt_train_acc, dt_val_acc, svm_train_acc, svm_val_acc

if __name__ == "__main__":
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/preprocessed_data-CISC_HTTPParams.csv"))
    df = load_and_preprocess_data(dataset_path)
    if df.empty:
        print("❌ Error: No valid data after preprocessing.")
    else:
        # Check for probability column
        if "probability" in df.columns:
            print("\n✅ Probability column detected. Using probability scores in modeling.")
            
            # Display probability distribution
            plt.figure(figsize=(8, 5))
            sns.histplot(df["probability"], bins=20, kde=True)
            plt.title("Distribution of Probability Scores")
            plt.xlabel("Probability")
            plt.ylabel("Count")
            plt.savefig(os.path.join(os.path.dirname(__file__), "../../results/probability_distribution.png"))
            plt.close()
            
            print(f"✅ Probability distribution chart saved to results/probability_distribution.png")
            
        # Train models    
        decision_tree, svm_model, dt_train_acc, dt_val_acc, svm_train_acc, svm_val_acc = train_models(df)
        print(f"\n✅ Final Results:")
        print(f"📊 Decision Tree Accuracy: Train={dt_train_acc:.2f}, Validation={dt_val_acc:.2f}")
        print(f"📊 SVM Accuracy: Train={svm_train_acc:.2f}, Validation={svm_val_acc:.2f}")
        print(f"✅ All models saved to models/model-3/ directory")
