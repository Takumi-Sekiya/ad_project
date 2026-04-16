import os
import argparse
import yaml
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
from sklearn.model_selection import StratifiedKFold
import torch

# 既存モジュールのインポート
from functions.data_handling import load_and_match_data, determine_target_canvas_size, create_dataset
from functions.utils import set_seed
from functions.data_loader import get_datasets
from functions.models import build_model
from functions.engine import run_training

# PyTorchの環境確認（GPUが認識されているか表示します）
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

def override_config(config: dict, roi: str, target: str):
    """
    コマンドライン引数で指定されたROIとターゲットに合わせてConfigを上書きする
    """
    if roi:
        print(f">> Config Override: Mask ROI -> {roi}")
        config['dataset_generation']['mask_name'] = roi
        config['dataset_generation']['path_templates']['mask'] = f'spm/norm/w{{subject}}_mask-{roi}.nii'

    if target:
        print(f">> Config Override: Target -> {target}")
        config['data']['target_variable'] = target
        config['dataset_generation']['columns_to_extract'] = [target]

    current_roi = config['dataset_generation']['mask_name']
    current_target = config['data']['target_variable']
    
    base_run_name = f"{config['model']['name']}_{current_roi}_to_{current_target}_cross_reg"
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

    allowed_diagnoses = gen_cfg.get('allowed_diagnoses', None)
    
    matched_df = load_and_match_data(
        processed_data_dir, 
        csv_path, 
        gen_cfg['path_templates'], 
        gen_cfg['columns_to_extract'],
        allowed_diagnoses=allowed_diagnoses
    )
    
    if matched_df.empty:
        raise ValueError("処理対象データが見つかりませんでした。")
    
    target_var = config['data']['target_variable']
    matched_df = matched_df.dropna(subset=[target_var]).reset_index(drop=True)
    print(f"有効データ数: {len(matched_df)}名")

    target_canvas_size = None
    if gen_cfg['roi_processing_mode'] == 'crop_and_pad':
        target_canvas_size = determine_target_canvas_size(matched_df)
        print(f"決定したキャンバスサイズ: {target_canvas_size}")

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
    
    try:
        bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
    except ValueError:
        print("警告: データ数が少なすぎるか値が重複しており、qcut(5)に失敗しました。cut(5)を使用します。")
        bins = pd.cut(y, bins=5, labels=False, duplicates='drop')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config['training']['random_state'])
    
    gen_cfg = config['dataset_generation']
    csv_path = base_dir / gen_cfg['clinical_csv_path']
    metadata_path = csv_path.parent / "scaling_metadata.json"
    config['data']['metadata_path'] = str(metadata_path)

    output_dir = base_dir / gen_cfg['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, test_idx) in enumerate(skf.split(img_all, bins)):
        print(f"\n{'-'*20} Fold {fold+1} / 5 {'-'*20}")
        
        X_train, X_test = img_all[train_idx], img_all[test_idx]
        y_train_df = features_all.iloc[train_idx].reset_index(drop=True)
        y_test_df = features_all.iloc[test_idx].reset_index(drop=True)
        
        print(f"Train: {len(X_train)} / Test: {len(X_test)}")

        pickle_filename = gen_cfg['filename_template'].format(
            mask_name=gen_cfg['mask_name'],
            target_name=target_var
        ).replace('.pkl', f'_fold{fold}.pkl')
        
        output_pickle_file = output_dir / pickle_filename

        dataset_dict = {
            'img_train': X_train, 'features_train': y_train_df,
            'img_test': X_test, 'features_test': y_test_df,
            'config': config
        }

        with open(output_pickle_file, 'wb') as f:
            pickle.dump(dataset_dict, f)

        rel_path = Path(gen_cfg['output_dir']) / pickle_filename
        config['data']['pickle_path'] = str(rel_path)

        base_name = config.get('base_run_name', 'Experiment')
        config['run_name'] = f"{base_name}/cross{fold}"

        training_phase(config)

def training_phase(config: dict):
    mlflow.set_experiment(config['experiment_name'])
    
    with mlflow.start_run(run_name=config['run_name']):
        mlflow.log_params(config['dataset_generation'])
        mlflow.log_params(config['model'])
        mlflow.log_params(config['training'])
        mlflow.log_param("target_variable", config['data']['target_variable'])
        mlflow.log_param("fold_pickle", config['data']['pickle_path'])
        
        set_seed(config['environment']['seed'])

        train_ds, test_ds = get_datasets(config)

        # PyTorchのDatasetからinput_shapeを取得
        # train_ds[0] は (sample_dict, label) を返すので、[0] で sample_dict を取得
        sample_dict = train_ds[0][0]
        img_shape = sample_dict['img_input'].shape
        
        model = build_model(input_shape=img_shape, config=config)
        
        run_training(model, train_ds, test_ds, config)

def main(args):
    base_dir = Path(__file__).resolve().parent.parent
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config = override_config(config, args.roi, args.target)

    try:
        img_all, features_all = prepare_full_dataset(config, base_dir)
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