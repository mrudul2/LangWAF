from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model & vectorizer
model = joblib.load("waf_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")  # Load the same vectorizer used in training

feature_columns = ['method_GET', 'method_POST']  # Adjusted

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    method = data.get('method', 'GET')
    payload = data.get('payload', '')

    # Convert method to one-hot encoding
    method_features = {'method_GET': 0, 'method_POST': 0}
    if method in method_features:
        method_features[f'method_{method}'] = 1

    # Vectorize payload (must match training process)
    payload_vectorized = vectorizer.transform([payload]).toarray()

    # Combine method and payload features
    input_data = pd.DataFrame(payload_vectorized, columns=vectorizer.get_feature_names_out())
    input_data = input_data.assign(**method_features)

    # Predict risk score
    risk_score = model.predict_proba(input_data)[0][1] * 100

    return jsonify({"risk_score": round(risk_score, 2)})

if __name__ == '__main__':
    app.run(debug=True)
