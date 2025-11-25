# pipeline_steps.py
"""
パイプラインの各ステップの処理を定義するモジュール.
各関数は, 原則として一人の被験者データに対する処理を担当する.
"""

import config.config as cfg
from functions.utils import run_command
from pathlib import Path

def step_01_dicom_to_nifti(sub_id):
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
    output_fs_dir = cfg.FS_SUBJECTS_DIR / sub_id
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

def step_03_prepare_nifti(sub_id):
    """
    FreeSurfer出力からNIfTIデータを準備.
    """
    print(f"--- Step 3: Prepare NIfTI for: {sub_id} ---")

    fs_sub_dir = cfg.FS_SUBJECTS_DIR / sub_id / "mri"

    anat_dir = cfg.PROCESSED_DATA_DIR / sub_id / "anat"
    mask_dir = cfg.PROCESSED_DATA_DIR / sub_id / "mask" / "lobes"
    anat_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    t1w_mgz = fs_sub_dir / "orig.mgz"
    t1w_nii = anat_dir / f"{sub_id}_T1w.nii"

    if t1w_mgz.exists() and not t1w_nii.exists():
        print(f"Converting orig.mgz to {t1w_nii}")
        cmd_t1w = f"mri_convert {t1w_mgz} {t1w_nii}"
        run_command(cmd_t1w)
    elif t1w_nii.exists():
        print(f"T1w NIfTI already exists for {sub_id}, skipping conversion.")
    
    aseg_mgz = fs_sub_dir / "aparc+aseg.mgz"

    if aseg_mgz.exists():
        for roi in cfg.LOBES:
            roi_name = roi['name']
            mask_nii = mask_dir / f"{sub_id}_mask-{roi_name}.nii"
            temp_mask_nii = mask_dir / f"{sub_id}_mask-{roi_name}_TEMP.nii"

            if not mask_nii.exists():
                print(f"Creating mask for ROI: {roi_name}")
                match_str = " ".join(map(str, roi["labels"]))
                cmd_mask = (
                    f"bash -c \""
                    f"mri_binarize --i {aseg_mgz} --match {match_str} --o {temp_mask_nii} && "
                    f"mri_convert -odt uchar {temp_mask_nii} {mask_nii} && "
                    f"rm {temp_mask_nii}\""
                )
                run_command(cmd_mask)
            else:
                print(f"Mask for ROI {roi_name} already exists for {sub_id}, skipping.")
    else:
        print(f"aparc+aseg.mgz does not exist for {sub_id}, cannot create ROI masks.")

def step_03b_prepare_nifti_roi_masks(sub_id):
    """
    FreeSurfer出力からNIfTIデータを準備.
    特定のROIに対してマスクを作成.
    """
    print(f"--- Step 3: Prepare NIfTI for: {sub_id} ---")

    fs_sub_dir = cfg.FS_SUBJECTS_DIR / sub_id / "mri"

    mask_dir = cfg.PROCESSED_DATA_DIR / sub_id / "mask" / "rois"
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    aseg_mgz = fs_sub_dir / "aparc+aseg.mgz"

    if aseg_mgz.exists():
        for roi in cfg.ROIS:
            roi_name = roi['name']
            mask_nii = mask_dir / f"{sub_id}_mask-{roi_name}.nii"
            temp_mask_nii = mask_dir / f"{sub_id}_mask-{roi_name}_TEMP.nii"

            if not mask_nii.exists():
                print(f"Creating mask for ROI: {roi_name}")
                match_str = " ".join(map(str, roi["labels"]))
                cmd_mask = (
                    f"bash -c \""
                    f"mri_binarize --i {aseg_mgz} --match {match_str} --o {temp_mask_nii} && "
                    f"mri_convert -odt uchar {temp_mask_nii} {mask_nii} && "
                    f"rm {temp_mask_nii}\""
                )
                run_command(cmd_mask)
            else:
                print(f"Mask for ROI {roi_name} already exists for {sub_id}, skipping.")
    else:
        print(f"aparc+aseg.mgz does not exist for {sub_id}, cannot create ROI masks.")

def step_03c_prepare_nifti_lobe_masks(sub_id):
    """
    FreeSurfer出力からNIfTIデータを準備.
    特定の脳葉に対してマスクを作成.
    """
    print(f"--- Step 3: Prepare NIfTI for: {sub_id} ---")

    fs_sub_dir = cfg.FS_SUBJECTS_DIR / sub_id / "mri"

    mask_dir = cfg.PROCESSED_DATA_DIR / sub_id / "mask" / "lobes"
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    aseg_mgz = fs_sub_dir / "aparc+aseg.mgz"

    if aseg_mgz.exists():
        for roi in cfg.LOBES:
            roi_name = roi['name']
            mask_nii = mask_dir / f"{sub_id}_mask-{roi_name}.nii"
            temp_mask_nii = mask_dir / f"{sub_id}_mask-{roi_name}_TEMP.nii"

            if not mask_nii.exists():
                print(f"Creating mask for ROI: {roi_name}")
                match_str = " ".join(map(str, roi["labels"]))
                cmd_mask = (
                    f"bash -c \""
                    f"mri_binarize --i {aseg_mgz} --match {match_str} --o {temp_mask_nii} && "
                    f"mri_convert -odt uchar {temp_mask_nii} {mask_nii} && "
                    f"rm {temp_mask_nii}\""
                )
                run_command(cmd_mask)
            else:
                print(f"Mask for ROI {roi_name} already exists for {sub_id}, skipping.")
    else:
        print(f"aparc+aseg.mgz does not exist for {sub_id}, cannot create ROI masks.")

def step_04_segment_hippocampus(sub_id):
    """
    FreeSurferの海馬小領域セグメンテーション (segmentHA_T1.sh) を実行.
    Step 2 (recon-all) が完了していることが前提.
    """
    print(f"--- Step 4: Hippocampal Subfield Segmentation for: {sub_id} ---")

    recon_all_done_file = cfg.FS_SUBJECTS_DIR / sub_id / "scripts" / "recon-all.done"
    if not recon_all_done_file.exists():
        print(f"recon-all not completed for {sub_id}, cannot run hippocampal segmentation.")
        return
    
    output_check_file = cfg.FS_SUBJECTS_DIR / sub_id / "mri" / "lh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz"
    if output_check_file.exists():
        print(f"Hippocampal segmentation already completed for {sub_id}, skipping.")
        return

    log_dir = cfg.FS_SUBJECTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{sub_id}.segmentHA_T1.log"

    setup_cmd = f"export FREESURFER_HOME={cfg.FREESURFER_HOME} && source $FREESURFER_HOME/SetUpFreeSurfer.sh"
    hippo_cmd = f"segmentHA_T1.sh {sub_id}"

    full_cmd = (
        f"bash -c \""
        f"{setup_cmd} && "
        f"export SUBJECTS_DIR={cfg.FS_SUBJECTS_DIR} && "
        f"{hippo_cmd} "
        f"\""
    )

    if not run_command(full_cmd, log_file=log_file):
        print(f"Failed to run hippocampal segmentation for {sub_id}. Check log: {log_file}")

def step_05_extract_subfield_masks(sub_id):
    """
    Step 4 で生成した海馬小領域ラベルから, 左右を統合したバイナリ NIfTI マスクを抽出.
    """
    print(f"--- Step 5: Extract Hippocampal Subfield Masks for: {sub_id} ---")

    fs_mri_dir = cfg.FS_SUBJECTS_DIR / sub_id / "mri"
    lh_sf_mgz = fs_mri_dir / "lh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz" # 実際に出力されたファイル名に合わせる
    rh_sf_mgz = fs_mri_dir / "rh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz" # 実際に出力されたファイル名に合わせる

    if not (lh_sf_mgz.exists() and rh_sf_mgz.exists()):
        print(f"Hippocampal subfield label files do not exist for {sub_id}, cannot extract masks.")
        return
    
    mask_hipp_dir = cfg.PROCESSED_DATA_DIR / sub_id / "mask" / "hippocampal_subfields"
    mask_hipp_dir.mkdir(parents=True, exist_ok=True)

    setup_cmd = (
        f"export FREESURFER_HOME={cfg.FREESURFER_HOME} && "
        f"source $FREESURFER_HOME/SetUpFreeSurfer.sh && "
        f"export SUBJECTS_DIR={cfg.FS_SUBJECTS_DIR}"
    )
    
    for roi in cfg.HIPPOCAMPAL_SUBFIELDS:
        roi_name = roi['name']
        mask_nii = mask_hipp_dir / f"{sub_id}_mask-{roi_name}.nii"
        temp_lh_nii = mask_hipp_dir / f"{sub_id}_mask-{roi_name}_TEMP_LH.nii"
        temp_merged_nii = mask_hipp_dir / f"{sub_id}_mask-{roi_name}_TEMP_MERGED.nii"

        for f in [temp_lh_nii, temp_merged_nii]:
            if f.exists(): f.unlink()

        match_str = " ".join(map(str, roi['labels']))



        # 1. LHマスク作成 (Temp LH)
        cmd_lh = (
            f"bash -c \"{setup_cmd} && "
            f"mri_binarize --i {lh_sf_mgz} --match {match_str} --o {temp_lh_nii}\""
        )
        if not run_command(cmd_lh):
            print(f"failed to create LH mask for ROI: {roi_name}")
            continue

        # 2. RHマスクとマージ (Temp Merged) -> 3. 軽量化して最終出力 -> 4. 一時ファイル削除
        cmd_merge_convert = (
            f"bash -c \"{setup_cmd} && "
            f"mri_binarize --i {rh_sf_mgz} --match {match_str} --merge {temp_lh_nii} --o {temp_merged_nii} && "
            f"mri_convert -odt uchar {temp_merged_nii} {mask_nii} && "
            f"rm {temp_lh_nii} {temp_merged_nii}\""
        )

        if not run_command(cmd_merge_convert):
            print(f"failed to create merged mask for ROI: {roi_name}")
            # 失敗時はゴミ掃除
            if temp_lh_nii.exists(): temp_lh_nii.unlink()
            if temp_merged_nii.exists(): temp_merged_nii.unlink()
            continue