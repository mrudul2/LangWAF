from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

import logging
import json
import os
import sys
import subprocess
import threading
import time
import traceback
import pandas as pd


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("langwaf_server.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("langwaf")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'langwaf-secure-key'
CORS(app)  # Allow all origins

# Track model execution status
model_status = {
    "model1": {"status": "idle", "progress": 0, "message": "Ready to execute"},
    "model2": {"status": "idle", "progress": 0, "message": "Ready to execute"},
    "model3": {"status": "idle", "progress": 0, "message": "Ready to execute"},
    "pipeline": {"status": "idle", "progress": 0, "message": "Ready to execute"}
}

# Lock for thread-safe status updates
status_lock = threading.Lock()

# Store execution results
execution_results = {
    "model1": {"execution_time": 0, "completed_at": "", "status": ""},
    "model2": {"execution_time": 0, "completed_at": "", "status": ""},
    "model3": {"execution_time": 0, "completed_at": "", "status": ""},
    "pipeline": {"execution_time": 0, "completed_at": "", "status": ""}
}

# Lock for thread-safe results updates
results_lock = threading.Lock()


def update_status(model, status, progress=None, message=None):
    """Update the status of a model or the pipeline"""
    with status_lock:
        model_status[model]["status"] = status
        if progress is not None:
            model_status[model]["progress"] = progress
        if message is not None:
            model_status[model]["message"] = message
        logger.info(f"Model {model} status updated: {status}, {progress}%, {message}")


def update_result(model, execution_time, status="completed"):
    """Update the execution results"""
    with results_lock:
        execution_results[model] = {
            "execution_time": execution_time,
            "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status
        }


def run_model1():
    """Run Model 1 (Base Classification)"""
    try:
        update_status("model1", "running", 0, "Starting Model 1 execution...")
        
        # Get the path to run.py
        model_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   "models", "model-1", "run.py")
        
        # Start time for performance measurement
        start_time = time.perf_counter()
        
        # Update status before execution
        update_status("model1", "running", 25, "Executing base classification model...")
        
        # Run as subprocess
        process = subprocess.run([sys.executable, model_script], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)
        
        # End time and calculate execution time
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        if process.returncode != 0:
            error_msg = f"Error in Model 1: {process.stderr}"
            logger.error(error_msg)
            update_status("model1", "error", 0, error_msg[:100])
            update_result("model1", execution_time, "error")
            return False
        
        # Store results
        update_result("model1", execution_time)
        
        update_status("model1", "completed", 100, f"Model 1 completed in {execution_time:.2f} seconds")
        return True
    except Exception as e:
        error_msg = f"Error in Model 1: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        update_status("model1", "error", 0, f"Error: {str(e)}")
        update_result("model1", 0, "error")
        return False


def run_model2():
    """Run Model 2 (Language Detection)"""
    try:
        update_status("model2", "running", 0, "Starting Model 2 execution...")
        
        # Get the path to language_detection.py
        model_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   "models", "model-2", "language_detection.py")
        
        # Start time for performance measurement
        start_time = time.perf_counter()
        
        # Update status before execution
        update_status("model2", "running", 25, "Executing language detection model...")
        
        # Run as subprocess
        process = subprocess.run([sys.executable, model_script], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)
        
        # End time and calculate execution time
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        if process.returncode != 0:
            error_msg = f"Error in Model 2: {process.stderr}"
            logger.error(error_msg)
            update_status("model2", "error", 0, error_msg[:100])
            update_result("model2", execution_time, "error")
            return False
        
        # Store results
        update_result("model2", execution_time)
        
        update_status("model2", "completed", 100, f"Model 2 completed in {execution_time:.2f} seconds")
        return True
    except Exception as e:
        error_msg = f"Error in Model 2: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        update_status("model2", "error", 0, f"Error: {str(e)}")
        update_result("model2", 0, "error")
        return False


def run_model3():
    """Run Model 3 (SVM Classification)"""
    try:
        update_status("model3", "running", 0, "Starting Model 3 execution...")
        
        # Get the path to run_svm.py
        model_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   "models", "model-3", "run_svm.py")
        
        # Start time for performance measurement
        start_time = time.perf_counter()
        
        # Update status before execution
        update_status("model3", "running", 25, "Executing SVM classification model...")
        
        # Run as subprocess
        process = subprocess.run([sys.executable, model_script], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)
        
        # End time and calculate execution time
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        if process.returncode != 0:
            error_msg = f"Error in Model 3: {process.stderr}"
            logger.error(error_msg)
            update_status("model3", "error", 0, error_msg[:100])
            update_result("model3", execution_time, "error")
            return False
        
        # Store results
        update_result("model3", execution_time)
        
        update_status("model3", "completed", 100, f"Model 3 completed in {execution_time:.2f} seconds")
        return True
    except Exception as e:
        error_msg = f"Error in Model 3: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        update_status("model3", "error", 0, f"Error: {str(e)}")
        update_result("model3", 0, "error")
        return False


def run_pipeline():
    """Run all models in sequence"""
    try:
        # Ensure pipeline status is set to running at the start
        update_status("pipeline", "running", 0, "Starting pipeline execution...")
        pipeline_start = time.perf_counter()
        
        # Run Model 1
        update_status("pipeline", "running", 10, "Running Model 1...")
        if not run_model1():
            update_status("pipeline", "error", 30, "Pipeline failed at Model 1")
            update_result("pipeline", time.perf_counter() - pipeline_start, "error")
            return False
        
        update_status("pipeline", "running", 33, "Model 1 completed. Running Model 2...")
        # Run Model 2
        if not run_model2():
            update_status("pipeline", "error", 60, "Pipeline failed at Model 2")
            update_result("pipeline", time.perf_counter() - pipeline_start, "error")
            return False
        
        update_status("pipeline", "running", 66, "Model 2 completed. Running Model 3...")
        # Run Model 3
        if not run_model3():
            update_status("pipeline", "error", 90, "Pipeline failed at Model 3")
            update_result("pipeline", time.perf_counter() - pipeline_start, "error")
            return False
        
        # Calculate total execution time
        pipeline_end = time.perf_counter()
        pipeline_time = pipeline_end - pipeline_start
        
        # Store combined results
        with results_lock:
            execution_results["pipeline"] = {
                "execution_time": pipeline_time,
                "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model1_time": execution_results["model1"].get("execution_time", 0),
                "model2_time": execution_results["model2"].get("execution_time", 0),
                "model3_time": execution_results["model3"].get("execution_time", 0),
                "status": "completed"
            }
        
        update_status("pipeline", "completed", 100, f"Pipeline completed in {pipeline_time:.2f} seconds")
        return True
    except Exception as e:
        error_msg = f"Error in pipeline: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        update_status("pipeline", "error", 0, f"Error: {str(e)}")
        update_result("pipeline", 0, "error")
        return False


@app.route('/')
def index():
    """Render main page"""
    # Get latest results
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    available_results = []
    
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.endswith('.json') or file.endswith('.csv'):
                file_path = os.path.join(results_dir, file)
                file_stats = os.stat(file_path)
                file_size = file_stats.st_size / 1024  # KB
                file_date = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                available_results.append({
                    'name': file,
                    'path': file_path,
                    'size': f"{file_size:.2f} KB",
                    'date': file_date
                })
    
    # Sort by date (newest first)
    available_results.sort(key=lambda x: x['date'], reverse=True)
    
    return render_template('index.html', 
                          results=available_results[:10],  # Show only 10 most recent results
                          model_status=model_status,
                          execution_results=execution_results)


@app.route('/run/<model>')
def run_model(model):
    """API endpoint to run a model or the pipeline"""
    logger.info(f"Received request to run {model}")
    
    # Check if model is already running
    with status_lock:
        if model_status.get(model, {}).get("status") == "running":
            return jsonify({"status": "already_running", "message": f"{model} is already running"})
        
        # Immediately set status to running to prevent race conditions
        model_status[model]["status"] = "running"
        model_status[model]["progress"] = 0
        model_status[model]["message"] = f"Starting {model} execution..."
    
    logger.info(f"Starting {model} in a separate thread")
    
    # Run in a separate thread based on the model requested
    if model == "model1":
        thread = threading.Thread(target=run_model1)
        thread.daemon = True
        thread.start()
    elif model == "model2":
        thread = threading.Thread(target=run_model2)
        thread.daemon = True
        thread.start()
    elif model == "model3":
        thread = threading.Thread(target=run_model3)
        thread.daemon = True
        thread.start()
    elif model == "pipeline":
        thread = threading.Thread(target=run_pipeline)
        thread.daemon = True
        thread.start()
    else:
        with status_lock:
            model_status[model]["status"] = "idle"  # Reset if invalid model
        return jsonify({"status": "error", "message": f"Unknown model {model}"})
    
    return jsonify({"status": "started", "message": f"{model} execution started"})


@app.route('/status/<model>')
def get_status(model):
    """API endpoint to get the status of a model or the pipeline"""
    with status_lock:
        if model in model_status:
            current_status = dict(model_status[model])  # Create a copy to avoid race conditions
            return jsonify(current_status)
        return jsonify({"status": "unknown", "progress": 0, "message": f"Unknown model {model}"})


@app.route('/results')
def get_results():
    """API endpoint to get the results of model executions"""
    with results_lock:
        current_results = json.loads(json.dumps(execution_results))  # Deep copy to avoid race conditions
    return jsonify(current_results)


@app.route('/files')
def get_files():
    """API endpoint to get available result files"""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    available_results = []
    
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.endswith('.json') or file.endswith('.csv'):
                file_path = os.path.join(results_dir, file)
                file_stats = os.stat(file_path)
                file_size = file_stats.st_size / 1024  # KB
                file_date = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                available_results.append({
                    'name': file,
                    'path': file_path,
                    'size': f"{file_size:.2f} KB",
                    'date': file_date
                })
    
    # Sort by date (newest first)
    available_results.sort(key=lambda x: x['date'], reverse=True)
    
    return jsonify(available_results)


@app.route('/file/<filename>')
def get_file_content(filename):
    """API endpoint to get the content of a result file"""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    file_path = os.path.join(results_dir, filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
        
    # For security, check that the path is within the results directory
    if not os.path.realpath(file_path).startswith(os.path.realpath(results_dir)):
        return jsonify({"error": "Access denied"}), 403
    
    try:
        if filename.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as file:
                return jsonify(json.load(file))
        elif filename.endswith('.csv'):
            # For CSV, return the raw text
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            return jsonify({"error": "Unsupported file type"}), 400
    except Exception as e:
        logger.error(f"Error reading file {filename}: {str(e)}")
        return jsonify({"error": f"Error reading file: {str(e)}"}), 500


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
    # Create necessary directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Initialize execution results with valid data structures
    for model in ['model1', 'model2', 'model3', 'pipeline']:
        if not execution_results.get(model):
            execution_results[model] = {"execution_time": 0, "completed_at": "", "status": ""}

    logger.info("Starting LangWAF server...")

    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
