import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import modules
sys.path.append(str(Path(__file__).parent))

from train import run_loso_training

def run_sanity_check():
    """
    Runs a minimal training session to verify that the pipeline is working correctly.
    Uses only 2 patients and 1 epoch per phase.
    """
    print("Starting Sanity Check (Minimal Training)...")
    
    # Path to the data
    data_dir = "../data/processed/siemens-net-data"
    
    # Check if data exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found.")
        return

    # In a real sanity check, we might want to temporarily mock the patient list
    # but for simplicity, we'll just run the training function with minimal epochs.
    patient_dirs = os.listdir(data_dir)
    test_patients = sorted([d for d in patient_dirs if os.path.isdir(os.path.join(data_dir, d))])[:2]
    
    print(f"This script will run LOSO training with patients: {test_patients}")
    print(" - 1 epoch for Phase 1 (Head)")
    print(" - 1 epoch for Phase 2 (Fine-tuning)")
    print(" - Results will be saved in 'sanity_check_results' directory.")
    
    # We call run_loso_training with minimal epochs and minimal patients
    run_loso_training(
        root_dir=data_dir,
        num_epochs_head=1,
        num_epochs_full=1,
        results_dir="sanity_check_results",
        patient_list=test_patients
    )

if __name__ == "__main__":
    # To run this script, uncomment the line below:
    run_sanity_check()
    print("Sanity check script ready. Uncomment 'run_sanity_check()' in the code to execute.")
