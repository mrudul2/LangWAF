from flask import Flask, jsonify
from flask_cors import CORS
import json
import os

import pandas as pd

app = Flask(__name__)
CORS(app)  # Allow all origins

@app.route('/api/model-results')
def get_model_results():
    try:
        # Use a relative path to navigate from src/ to results/
        file_path = os.path.join(os.path.dirname(__file__), '../results/model3_results.json')
        
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/model1-comparison-summary')
def get_model1_comparison_summary():
    try:
        file_path = os.path.join(os.path.dirname(__file__), '../results/model1_comparison_summary.json')
        with open(file_path, 'r') as file:
            data = json.load(file)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/model2-language-summary', methods=['GET'])
def model2_language_summary():
    df = pd.read_csv('../data/model-2_data_with_languages-CISC_HTTPParams.csv')

    summary = {}

    for _, row in df.iterrows():
        lang = str(row['detected_language']).strip().lower()
        classification = str(row['classification']).strip().lower()

        if not lang or not classification:
            continue

        if lang not in summary:
            summary[lang] = {"safe": 0, "unsafe": 0}

        if classification == "safe":
            summary[lang]["safe"] += 1
        else:
            summary[lang]["unsafe"] += 1

    return jsonify(summary)



if __name__ == '__main__':
    app.run(debug=True)
