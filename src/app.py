from flask import Flask, jsonify
from flask_cors import CORS
import json
import os

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

if __name__ == '__main__':
    app.run(debug=True)
