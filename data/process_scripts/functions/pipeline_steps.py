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

def step_02_run_recon_all(sub_id):
    """
    FreeSurferのrecon-allを実行.
    """
    print(f"--- Step 2: FreeSurfer recon-all for: {sub_id} ---")

    input_nifti = cfg.BIDS_NIFTI_DIR / sub_id / f"{sub_id}_T1w.nii"
    output_fs_dir = cfg.FREESURFER_DIR / sub_id
    done_file = output_fs_dir / "scripts" / "recon-all.done"

    if done_file.exists():
        print(f"recon-all already completed for {sub_id}, skipping.")
        return
    
    if not input_nifti.exists():
        print(f"Input NIfTI does not exist for {sub_id}: {input_nifti}")
        return

    log_dir = cfg.FS_SUBJECTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{sub_id}recon-all.log"

    setup_cmd = f"export FREESURFER_HOME={cfg.FREESURFER_HOME} && source $FREESURFER_HOME/SetUpFreeSurfer.sh"
    recon_cmd = f"recon-all -s {sub_id} -i {input_nifti} -all"

    full_cmd = (
        f"bash -c \""
        f"{setup_cmd} && "
        f"export SUBJECTS_DIR={cfg.FS_SUBJECTS_DIR} && "
        f"{recon_cmd} "
        f"\""
    )

    if not run_command(full_cmd, log_file=log_file):
        print(f"Failed to run recon-all for {sub_id}. Check log: {log_file}")

