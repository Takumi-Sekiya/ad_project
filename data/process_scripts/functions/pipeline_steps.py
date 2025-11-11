# pipeline_steps.py
"""
パイプラインの各ステップの処理を定義するモジュール.
各関数は, 原則として一人の被験者データに対する処理を担当する.
"""

import config.config as cfg
from functions.utils import run_command
from pathlib import Path

def step1_dicom_to_nifti(sub_id):
    """
    DICOMデータをNIfTI形式に変換.
    """
    print(f"--- Step 1: DICOM -> NIfTI for: {sub_id} ---")

    dicom_in_path = cfg.RAW_DATA_DIR / sub_id 
    nifti_out_path = cfg.BIDS_NIFTI_DIR / sub_id
    output_nifti = nifti_out_path / f"{sub_id}_T1w.nii"

    nifti_out_path.mkdir(parents=True, exist_ok=True)

    if output_nifti.exists():
        print(f"NIfTI file already exists for {sub_id}, skipping conversion.")
        return
    
    if not dicom_in_path.exists():
        print(f"DICOM input path does not exist for {sub_id}: {dicom_in_path}")
        return
    
    cmd = f"dcm2niix -o {nifti_out_path} -f {sub_id}_T1w -b y -z n {dicom_in_path}"

    if not run_command(cmd):
        print(f"Failed to convert DICOM to NIfTI for {sub_id}")

