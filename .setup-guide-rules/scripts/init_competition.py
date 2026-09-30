import os
import sys
import shutil
import subprocess
import argparse

def init_competition(competition_name, target_dir):
    print(f"Initializing Kaggle Competition: {competition_name}")
    print(f"Target Directory: {target_dir}")
    
    os.makedirs(target_dir, exist_ok=True)
    
    # 1. Download Data
    data_dir = os.path.join(target_dir, "dataset")
    os.makedirs(data_dir, exist_ok=True)
    
    print("Downloading data using Kaggle CLI...")
    try:
        subprocess.check_call(["kaggle", "competitions", "download", "-c", competition_name, "-p", data_dir])
        
        # Unzip if it's a zip file
        zip_file = os.path.join(data_dir, f"{competition_name}.zip")
        if os.path.exists(zip_file):
            print("Unzipping downloaded data...")
            import zipfile
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            os.remove(zip_file)
            print("Successfully extracted dataset.")
    except subprocess.CalledProcessError:
         print("Warning: Failed to download data automatically via CLI.")
         print("Ensure kaggle CLI is installed and ~/.kaggle/kaggle.json is configured.")
    except Exception as e:
        print(f"Error: {e}")
        
    # 2. Copy Templates and Lab Book
    print("Setting up notebook and tracking environment...")
    setup_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    templates_dir = os.path.join(setup_dir, "templates")
    target_notebooks_dir = os.path.join(target_dir, "notebooks")
    
    os.makedirs(target_notebooks_dir, exist_ok=True)
    
    if os.path.exists(templates_dir):
        files_to_copy = ["Experiment_Template.ipynb", "AI_LAB_BOOK.md"]
        for f in files_to_copy:
            src = os.path.join(templates_dir, f)
            if os.path.exists(src):
                # We place the Lab Book at the root of the competition, and the notebook inside notebooks/
                if f.endswith(".md"):
                    dst = os.path.join(target_dir, f)
                else:
                    dst = os.path.join(target_notebooks_dir, f)
                shutil.copy2(src, dst)
                print(f"Copied {f}")
            else:
                print(f"Warning: {f} not found in {templates_dir}")
    else:
        print(f"Warning: Templates directory not found at {templates_dir}")
        
    print(f"\nSuccessfully initialized {competition_name} at {target_dir}")
    print("Instructions:")
    print("1. Duplicate notebooks/Experiment_Template.ipynb and rename it (e.g., Exp01_Baseline.ipynb).")
    print("2. Work on the experiment end-to-end.")
    print("3. Log your CV and Leaderboard scores in AI_LAB_BOOK.md.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize a Kaggle competition project.")
    parser.add_argument("competition_name", help="The Kaggle competition name (e.g., titanic)")
    parser.add_argument("target_dir", default=".", nargs="?", help="The local directory to initialize the project in (default: current directory)")
    args = parser.parse_args()
    
    init_competition(args.competition_name, args.target_dir)
