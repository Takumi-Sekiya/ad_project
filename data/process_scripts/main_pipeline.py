# main_pipeline.py
"""
MRI画像前処理パイプラインのメインスクリプト.

(1) config.py でパラメータを設定
(2) condif.py で実行するステップのフラグをTrueにする
(3) このスクリプトを実行 python data/process_scripts/main_pipeline.py
"""

import config.config as cfg
import functions.pipeline_steps as steps
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def find_subjects(raw_data_dir, prefixes):
    """
    指定されたプレフィックスに基づいて被験者IDを収集する.
    """
    subject_ids = set()
    for prefix in prefixes:
        for path in raw_data_dir.glob(prefix):
            if path.is_dir():
                subject_ids.add(path.name)
    
    subject_ids.discard(".")
    subject_ids.discard("..")

    return sorted(list(subject_ids))

def run_step_parallel(step_function, subject_ids, n_cores):
    """
    指定されたステップ関数を, 被験者リストに対して並列実行する.
    """
    with ThreadPoolExecutor(max_workers=n_cores) as executor:
        futures = {
            executor.submit(step_function, sub_id): sub_id
            for sub_id in subject_ids
        }

        for future in as_completed(futures):
            sub_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing subject {sub_id}: {e}")

def main():
    """
    パイプラインの実行
    """
    print("Starting MRI preprocessing pipeline...")
    start_time = time.time()

    print("Collecting subject IDs...")
    subject_ids = find_subjects(cfg.RAW_DATA_DIR, cfg.SUBJECT_PREFIXES)
    if not subject_ids:
        print("No subjects found. Exiting.")
        return
    print(f"Found {len(subject_ids)} subjects: {subject_ids}")
    print(f"Using {cfg.N_CORES} cores for parallel processing.")

    # --- Step 1: DICOM to NIfTI ---
    if cfg.STEP_FLAGS['run_step1_dicom_to_nifti']:
        print("\n===== Step 1: DICOM -> NIfTI 変換 =====")
        run_step_parallel(steps.step_01_dicom_to_nifti, subject_ids, cfg.N_CORES)

    # --- Step 2: Run FreeSurfer recon-all ---
    if cfg.STEP_FLAGS['run_step2_run_recon_all']:
        print("\n===== Step 2: FreeSurfer recon-all 実行 =====")
        run_step_parallel(steps.step_02_run_recon_all, subject_ids, cfg.N_CORES)

    # --- Step 3: NIfTIファイルの準備 (FS出力から) ---
    if cfg.STEP_FLAGS['run_step3_prepare_nifti']:
        print("\n===== Step 3: NIfTIファイル準備 (FS出力から) =====")
        run_step_parallel(steps.step_03_prepare_nifti, subject_ids, cfg.N_CORES)
    
    # --- Step 4: Hippocampal Subfield Segmentation ---
    if cfg.STEP_FLAGS['run_step4_segment_hippocampus']:
        print("\n===== Step 4: Hippocampal Subfield Segmentation =====")
        run_step_parallel(steps.step_04_segment_hippocampus, subject_ids, cfg.N_CORES)

    end_time = time.time()
    print(f"\nPipeline completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()