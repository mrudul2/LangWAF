import time
import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from train import compute_weighted_score

data_dir = os.path.join(os.path.dirname(__file__), "../../data")
results_dir = os.path.join(os.path.dirname(__file__), "../../results")

def load_model(model_path):
    """
    Load the trained SVM model

    Args:
        model_path: Path to the saved SVM model

    Returns:
        The loaded SVM model
    """
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def load_scaler(scaler_path):
    """
    Load the fitted StandardScaler

    Args:
        scaler_path: Path to the saved scaler

    Returns:
        The loaded scaler
    """
    try:
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded successfully from {scaler_path}")
        return scaler
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None


def preprocess_input_data(data_path, scaler):
    """
    Preprocess input data for prediction

    Args:
        data_path: Path to the input data CSV
        scaler: The StandardScaler to use for scaling features

    Returns:
        Preprocessed data ready for prediction
    """
    column_names = [
        "url", "label", "detected_language", "label_y", "is_unsafe", "payload_length",
        "num_special_chars", "url_depth", "probability", "lang_match", "precision_m1", "recall_m1",
        "fpr_m1", "fnr_m1", "confidence_m2", "detected_language_confidence"
    ]

    df = pd.read_csv(data_path, names=column_names, header=0, low_memory=False)

    # Fill NA values
    if "label" in df.columns:
        df["label"] = df["label"].fillna("unknown").astype(str)

    # Convert numeric columns
    numeric_columns = [
        "is_unsafe", "payload_length", "num_special_chars", "url_depth", "probability",
        "lang_match", "precision_m1", "recall_m1", "fpr_m1", "fnr_m1", "confidence_m2",
        "detected_language_confidence"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Handle language encoding
    if "detected_language" in df.columns:
        le = LabelEncoder()
        df["detected_language_encoded"] = le.fit_transform(df["detected_language"].astype(str))

    # Drop unnecessary columns
    df = df.drop(columns=["url", "label_y", "detected_language"], errors='ignore')

    # Compute weighted score
    df["weighted_score"] = df.apply(lambda row: compute_weighted_score(
        row["is_unsafe"], row["lang_match"], row["precision_m1"], row["recall_m1"],
        row["fpr_m1"], row["fnr_m1"], row["confidence_m2"], row["detected_language_confidence"],
        row["detected_language_encoded"]), axis=1)

    # Prepare features (drop label if it exists)
    X = df.drop(columns=["label"], errors='ignore')

    # Scale the features
    X_scaled = scaler.transform(X)

    return X_scaled, df


def predict(model, X):
    """
    Make predictions using the loaded model

    Args:
        model: The SVM model
        X: Preprocessed and scaled features

    Returns:
        Predictions
    """
    try:
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        return predictions, probabilities
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None, None


def save_predictions(df, predictions, probabilities, output_path):
    """
    Save predictions to a CSV file

    Args:
        df: Original dataframe
        predictions: Model predictions
        probabilities: Prediction probabilities
        output_path: Path to save predictions
    """
    results_df = df.copy()
    results_df['predicted_label'] = predictions

    # Add probability for each class
    class_names = model.classes_
    for i, class_name in enumerate(class_names):
        results_df[f'prob_{class_name}'] = probabilities[:, i]

    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    # Paths
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, "svm_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    # Check if model and scaler exist, otherwise inform user
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. You need to train the model first.")
        print("Run the following command to train:")
        print("python train.py")
        exit(1)

    if not os.path.exists(scaler_path):
        print(f"Scaler file not found at {scaler_path}. You need to train the model first.")
        exit(1)

    # Load model and scaler
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)

    if model is None or scaler is None:
        print("Failed to load model or scaler. Exiting.")
        exit(1)

    # Get input data path from user
    data_path = "models/model-3/preprocessed_data-CISC_HTTPParams.csv"
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        exit(1)

    # Preprocess data
    X_scaled, original_df = preprocess_input_data(data_path, scaler)

    # Make predictions
    start = time.perf_counter()
    predictions, probabilities = predict(model, X_scaled)
    end = time.perf_counter()
    print(f"Prediction time: {end - start:.4f} seconds")

    if predictions is not None:
        # Get output path from user
        output_path = os.path.join(results_dir, "predictions_svm.csv")

        # Save predictions
        save_predictions(original_df, predictions, probabilities, output_path)

        # Display sample of predictions
        print("\nSample predictions:")
        sample_size = min(5, len(predictions))
        for i in range(sample_size):
            print(f"Sample {i+1}: Predicted as '{predictions[i]}' with confidence {max(probabilities[i]):.4f}")
