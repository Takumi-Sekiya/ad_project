import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor



# --- パラメータ設定 ---
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / 'raw_data'
NIFTI_OUT_DIR = PROJECT_ROOT / 'derivatives' / 'bids_nifti'

SUBJECT_PREFIXES = ['YGT_*', 'SND_*']

MAX_WORKERS = max((os.cpu_count() - 4), 1)

def rename_problematic_files(subject_dicom_path: Path) -> list:
    """
    ディレクトリを探索し、「.」で始まるファイルの名前を一時的に変更

    Args:
        subject_dicom_path (Path): 対象被験者のDICOMルートフォルダ

    Returns:
        list: (元のパス, 一時的なパス) のタプルリスト
    """
    renamed_files = []
    for file_path in subject_dicom_path.rglob('*'):
        if file_path.is_file() and file_path.name.startswith('.'):
            #新しいファイル名を生成
            new_name = 'temp_' + file_path.name.lstrip('.')
            temp_path = file_path.with_name(new_name)

            #ファイル名を変更
            file_path.rename(temp_path)
            renamed_files.append((file_path, temp_path))

    return renamed_files

def restore_original_filenames(renamed_files: list):
    """
    名前の変更をしたファイル名をもとに戻す

    Args:
        renamed_files (list): (元のパス, 一時的なパス) のタプルリスト
    """
    if not renamed_files:
        return
    
    for original_path, temp_path in renamed_files:
        try:
            if temp_path.exists():
                temp_path.rename(original_path)
        except OSError as e:
            print(f"    Error restoring {temp_path}: {e}")

def process_subject(subject_id: str):
    """
    一人の被験者に対してDICOMからNIFTIへの変換を行う
    """
    print(f"Processing: {subject_id}")

    dicom_in_path = RAW_DATA_DIR / subject_id
    nifti_out_path = NIFTI_OUT_DIR / subject_id

    #出力ファイルが既にある場合はスキップ
    expected_nifti = nifti_out_path / f"{subject_id}_T1w.nii"
    if expected_nifti.exists():
        print(f"  -> Skipped: Output already exists for {subject_id}")
        return f"Skipped: {subject_id}"
    
    #出力ディレクトリを作成
    nifti_out_path.mkdir(parents=True, exist_ok=True)

    renamed_files = []
    try:
        #ファイル先頭にある「.」に対応
        renamed_files = rename_problematic_files(dicom_in_path)
        if renamed_files:
            print(f"  -> Renamed {len(renamed_files)} files starting with '.'")
        
        #dcm2niixコマンドの構築と実行
        cmd = [
            'dcm2niix',
            '-o', str(nifti_out_path),
            '-f', f"{subject_id}_T1w",
            '-b', 'y',
            '-z', 'n',
            str(dicom_in_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            print(f"  -> ERROR: dcm2niix failed for {subject_id}.")
            print(f"  -> Stderr: {result.stderr.strip()}")
            return f"Failed: {subject_id}"
        
        print(f"  -> Success: {subject_id} converted.")
        return f"Success: {subject_id}"
    
    except Exception as e:
        print(f"  -> An unexpected error occurred for {subject_id}: {e}")
        return f"Error: {subject_id}"
    
    finally:
        restore_original_filenames(renamed_files)
    
def main():
    """
    メイン関数
    """
    print("--- DICOM to NIfTI Conversion ---")

    all_subjects = []
    for prefix in SUBJECT_PREFIXES:
        all_subjects.extend(RAW_DATA_DIR.glob(prefix))

    subject_ids = sorted([path.name for path in all_subjects if path.is_dir()])

    if not subject_ids:
        print("No subjects found matching the specified prefixes. Exiting.")
        return
    
    print(f"Found {len(subject_ids)} potential subjects.")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_subject, subject_ids))

    success_count = sum(1 for r in results if r.startswith("Success"))
    skipped_count = sum(1 for r in results if r.startswith("Skipped"))
    failed_count = sum(1 for r in results if r.startswith("Failed") or r.startswith("Error"))
    
    print("\n--- Conversion Summary ---")
    print(f"Successfully converted: {success_count}")
    print(f"Skipped (already done): {skipped_count}")
    print(f"Failed or Errored:    {failed_count}")
    print("--------------------------")


if __name__ == "__main__":
    main()