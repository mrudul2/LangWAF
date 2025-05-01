import os
import time
import subprocess
import sys


def run_command(command, description):
    """
    Run a command and print execution time
    
    Args:
        command: Command to run
        description: Description of the command for output
    """
    print(f"\n{'=' * 80}")
    print(f"Running {description}...")
    print(f"{'=' * 80}")
    
    start_time = time.perf_counter()
    
    # Run the command and capture output
    process = subprocess.run(command, shell=True, text=True)
    
    if process.returncode != 0:
        print(f"Error running {description}. Exit code: {process.returncode}")
        return False
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    print(f"\n{'-' * 80}")
    print(f"{description} completed in {execution_time:.2f} seconds")
    print(f"{'-' * 80}")
    
    return True


def main():
    # Get project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Define model paths
    model1_path = os.path.join(project_root, "models", "model-1", "run.py")
    model2_path = os.path.join(project_root, "models", "model-2", "language_detection.py")
    model3_path = os.path.join(project_root, "models", "model-3", "run_svm.py")
    
    # Check if all model files exist
    for path, name in [(model1_path, "Model 1"), (model2_path, "Model 2"), (model3_path, "Model 3")]:
        if not os.path.exists(path):
            print(f"Error: {name} script not found at {path}")
            return
    
    # Record overall start time
    overall_start = time.perf_counter()
    
    # Run Model 1
    if not run_command(f"{sys.executable} {model1_path}", "Model 1 (Base Classification)"):
        print("Stopping execution due to error in Model 1")
        return
        
    # Run Model 2
    if not run_command(f"{sys.executable} {model2_path}", "Model 2 (Language Detection)"):
        print("Stopping execution due to error in Model 2")
        return
    
    # Run Model 3
    if not run_command(f"{sys.executable} {model3_path}", "Model 3 (SVM Classification)"):
        print("Stopping execution due to error in Model 3")
        return
    
    # Calculate and display overall execution time
    overall_end = time.perf_counter()
    overall_time = overall_end - overall_start
    
    print(f"\n{'=' * 80}")
    print(f"All models executed successfully!")
    print(f"Total execution time: {overall_time:.2f} seconds")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
