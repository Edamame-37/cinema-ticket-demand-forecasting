import os
import sys
import shutil

def copy_templates(target_dir):
    setup_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    templates_dir = os.path.join(setup_dir, "templates")
    
    if not os.path.exists(templates_dir):
        print(f"Error: Templates directory not found at {templates_dir}")
        sys.exit(1)
        
    os.makedirs(target_dir, exist_ok=True)
    
    files_to_copy = ["Experiment_Template.ipynb", "AI_LAB_BOOK.md"]
    for f in files_to_copy:
        src = os.path.join(templates_dir, f)
        if os.path.exists(src):
            dst = os.path.join(target_dir, f)
            shutil.copy2(src, dst)
            print(f"Copied {f} to {target_dir}")
            
if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    copy_templates(target)
