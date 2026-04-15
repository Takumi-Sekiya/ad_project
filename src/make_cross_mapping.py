import argparse
import yaml
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import nibabel as nib
from tqdm import tqdm

# 既存モジュールのインポート
from functions.data_handling import load_and_preprocess_mask, load_and_match_data, determine_target_canvas_size
from functions.utils import set_seed
from functions.data_loader import get_datasets
from functions.models import build_model
from functions.engine import run_training
from functions.image_preprocessing import crop_roi, pad_to_canvas, normalize_intensity

def create_dataset(df, mode, target_columns, canvas_shape=None):
    images, features = [], []
    print(f"データセットを生成中 (モード: {mode})...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        base_img = nib.load(row['base']).get_fdata().astype(np.float32)
        mask_img = load_and_preprocess_mask(row['mask'], threshold=100)
        roi_array = base_img * (mask_img > 0.5)

        if mode == 'crop_and_pad':
            if canvas_shape is None:
                raise ValueError("crop_and_padモードにはcanvas_shapeが必要です")
            cropped = crop_roi(roi_array)
            padded = pad_to_canvas(cropped, canvas_shape)
            processed_img = normalize_intensity(padded)
        elif mode == 'simple_mask':
            processed_img = normalize_intensity(roi_array)
        else:
            raise ValueError(f"未知のROI処理モードです: {mode}")
        
        images.append(processed_img)
        features.append(row[['subject_id'] + target_columns])

    img_array = np.array(images)[..., np.newaxis]
    features_df = pd.DataFrame(features).reset_index(drop=True)

    return img_array, features_df

def override_config(config: dict, roi: str, target: str):
    """
    コマンドライン引数で指定されたROIとターゲットに合わせてConfigを上書きする
    """
    if roi:
        print(f">> Config Override: Mask ROI -> {roi}")
        config['dataset_generation']['mask_name'] = roi
        # 安全策として、mask-{roi}.nii の形式を強制します
        config['dataset_generation']['path_templates']['mask'] = f'spm/norm/w{{subject}}_mask-{roi}.nii'

    if target:
        print(f">> Config Override: Target -> {target}")
        config['data']['target_variable'] = target
        config['dataset_generation']['columns_to_extract'] = [target]

    # ベースとなるRun Nameの設定 (後でcross{i}を付与するためにベース名を保持)
    current_roi = config['dataset_generation']['mask_name']
    current_target = config['data']['target_variable']
    
    # ここでは親ディレクトリ名としてベース名を設定
    base_run_name = f"{config['model']['name']}_{current_roi}_to_{current_target}_cross"
    config['base_run_name'] = base_run_name 
    
    return config

def prepare_full_dataset(config: dict, base_dir: Path):
    """
    全データを読み込み、前処理を行って画像配列と特徴量DFを返す。
    交差検証のループの前に一度だけ実行する。
    """
    print("\n=== Phase 1: 全データの読み込みと前処理を開始 ===")
    
    gen_cfg = config['dataset_generation']
    
    processed_data_dir = base_dir / gen_cfg['raw_data_base_dir']
    csv_path = base_dir / gen_cfg['clinical_csv_path']
    
    # 1. マッチング実行
    matched_df = load_and_match_data(
        processed_data_dir, 
        csv_path, 
        gen_cfg['path_templates'], 
        gen_cfg['columns_to_extract']
    )
    
    if matched_df.empty:
        raise ValueError("処理対象データが見つかりませんでした。")
    
    # ターゲットの欠損除去
    target_var = config['data']['target_variable']
    matched_df = matched_df.dropna(subset=[target_var]).reset_index(drop=True)
    print(f"有効データ数: {len(matched_df)}名")

    # 2. キャンバスサイズ決定
    target_canvas_size = None
    if gen_cfg['roi_processing_mode'] == 'crop_and_pad':
        target_canvas_size = determine_target_canvas_size(matched_df)
        print(f"決定したキャンバスサイズ: {target_canvas_size}")

    # 3. 画像生成 (全データ分)
    print("全画像のデータセット生成中...")
    img_all, features_all = create_dataset(
        matched_df,
        gen_cfg['roi_processing_mode'],
        gen_cfg['columns_to_extract'],
        target_canvas_size
    )
    
    return img_all, features_all

def run_cross_validation(img_all, features_all, config: dict, base_dir: Path):
    """
    5分割交差検証を実行するループ
    """
    print("\n=== Phase 2: 5分割交差検証 (5-fold CV) を開始 ===")

    target_var = config['data']['target_variable']
    y = features_all[target_var].values
    
    # 層化抽出のためのビン分割 (pd.qcutを使用)
    # データ数が少ない場合、q=5で分割できないことがあるため、エラーハンドリングまたはduplicate='drop'
    try:
        bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
    except ValueError:
        print("警告: データ数が少なすぎるか値が重複しており、qcut(5)に失敗しました。cut(5)を使用します。")
        bins = pd.cut(y, bins=5, labels=False, duplicates='drop')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config['training']['random_state'])

    # 交差検証ループ
    test_dfs = []
    for _, (_, test_idx) in enumerate(skf.split(img_all, bins)):
        y_test_df = features_all.iloc[test_idx].reset_index(drop=True)
        test_dfs.append(y_test_df)

    excel_path = f"data/processed/cross_validation_mapping_{config['dataset_generation']['mask_name']}_{target_var}.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        for i in range(5):
            test_dfs[i].to_excel(writer, sheet_name=f'Test_Data_Fold_{i+1}', index=False)


def main(args):
    base_dir = Path(__file__).resolve().parent.parent
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Configを動的に書き換え
    config = override_config(config, args.roi, args.target)

    try:
        # Phase 1: 全データの準備 (メモリ上に展開)
        img_all, features_all = prepare_full_dataset(config, base_dir)
        
        # Phase 2: 交差検証ループ実行
        run_cross_validation(img_all, features_all, config, base_dir)

    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='src/config/config.yaml', help='Path to the base config file')
    parser.add_argument('--roi', type=str, default=None, help='Override ROI name')
    parser.add_argument('--target', type=str, default=None, help='Override target variable')
    
    args = parser.parse_args()
    main(args)

# python src/make_cross_mapping.py --config src/config/config.yaml --roi gray-matter --target MMSE