import os
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / 'raw_data'
BIDS_NIFTI_DIR = PROJECT_ROOT / 'derivatives' / 'bids_nifti'

SUBJECT_PREFIXES = ['YGT_', 'SND_']

MAX_WORKERS = os.cpu_count() or 1

def convert_subject(subject_dir: Path) -> str:
    subject_id = subject_dir.name
    output_dir = BIDS_NIFTI_DIR / subject_id
    expected_nifti = output_dir / f"{subject_id}_T1w.nii"

    if expected_nifti.exists():
        return f"Skipped: {subject_id} (laready converted)"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    command = ["dcm2niix", "-o", str(output_dir), "-f", f"{subject_id}_T1w.nii", "-b", "y", "-z", "n", str(subject_dir)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        if expected_nifti.exists():
            return f"Success: {subject_id}"
        else:
            return f"Warning: {subject_id} (dcm2niix ran but output not found. Check logs.)\n{result.stdout}\n{result.stderr}"
        
    except FileNotFoundError:
        return "Error: 'dcm2niix' command not found. Is it installed and in your system's PATH?"
    except subprocess.CalledProcessError as e:
        return f"Failed: {subject_id}\n{e.stderr}"
    
def main():
    print("--- DICOM to NIfTI Conversion ---")
    print(f"Raw data source: {RAW_DATA_DIR}")
    print(f"Output directory: {BIDS_NIFTI_DIR}\n")

    subject_dirs_to_process = []
    for entry in RAW_DATA_DIR.iterdir():
        if entry.is_dir() and any(entry.name.startswith(prefix) for prefix in SUBJECT_PREFIXES):
            subject_dirs_to_process.append(entry)
    
    if not subject_dirs_to_process:
        print("No subjects found matching the specified prefixes. Exiting.")
        return
    
    print(f"Found {len(subject_dirs_to_process)} subjects. Starting conversion with {MAX_WORKERS} processes...")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(convert_subject, sub_dir) for sub_dir in subject_dirs_to_process]

        for future in as_completed(futures):
            try:
                message = future.result()
                print(message)
            except Exception as e:
                print(f"An error occurred in a worker process: {e}")

    print("\n--- All subjects processed. ---")


if __name__ == "__main__":
    main()