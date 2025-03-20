import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def train_models(df):
    X = df.drop(columns=["label"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    decision_tree = DecisionTreeClassifier(max_depth=5)
    decision_tree.fit(X_train, y_train)
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)
    
    dt_accuracy = decision_tree.score(X_test, y_test)
    svm_accuracy = svm_model.score(X_test, y_test)
    
    return decision_tree, svm_model, dt_accuracy, svm_accuracy

def compute_weighted_score(is_unsafe, lang_match, precision_m1, recall_m1, fpr_m1, fnr_m1, confidence_m2):
    W1 = ((precision_m1 + recall_m1) / 2) * (1 - fpr_m1)
    W2 = (1 - fnr_m1) * confidence_m2
    final_score = (W1 * is_unsafe) + (W2 * lang_match)
    return final_score

def classify_request(final_score):
    if final_score > 0.8:
        return "Bypassable ✅"
    elif 0.5 <= final_score <= 0.8:
        return "Further Inspection ⚠️"
    else:
        return "Blocked ❌"

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), "../../data/preprocessed_data.csv")
    df = pd.read_csv(dataset_path)
    decision_tree, svm_model, dt_accuracy, svm_accuracy = train_models(df)
    
    print(f"Decision Tree Accuracy: {dt_accuracy:.2f}")
    print(f"SVM Accuracy: {svm_accuracy:.2f}")
    
    # Example request classification
    example_score = compute_weighted_score(is_unsafe=1, lang_match=0, precision_m1=0.85, recall_m1=0.90,
                                           fpr_m1=0.10, fnr_m1=0.05, confidence_m2=0.95)
    decision = classify_request(example_score)
    print(f"Example Decision: {decision}")
