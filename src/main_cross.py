import os
import argparse
import yaml
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

# 既存モジュールのインポート
from functions.data_handling import load_and_match_data, determine_target_canvas_size, create_dataset
from functions.utils import set_seed
from functions.data_loader import get_datasets
from functions.models import build_model
from functions.engine import run_training

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

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
    
    # 出力先ディレクトリの準備
    gen_cfg = config['dataset_generation']
    output_dir = base_dir / gen_cfg['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    # 交差検証ループ
    for fold, (train_idx, test_idx) in enumerate(skf.split(img_all, bins)):
        print(f"\n{'-'*20} Fold {fold+1} / 5 {'-'*20}")
        
        # データの分割
        X_train, X_test = img_all[train_idx], img_all[test_idx]
        y_train_df = features_all.iloc[train_idx].reset_index(drop=True)
        y_test_df = features_all.iloc[test_idx].reset_index(drop=True)
        
        print(f"Train: {len(X_train)} / Test: {len(X_test)}")

        # Fold専用のPickleファイル名を作成
        # 例: dataset_brain-stem_gm_atrophy_fold0.pkl
        pickle_filename = gen_cfg['filename_template'].format(
            mask_name=gen_cfg['mask_name'],
            target_name=target_var
        ).replace('.pkl', f'_fold{fold}.pkl')
        
        output_pickle_file = output_dir / pickle_filename

        # data_loaderが期待する形式で辞書を作成
        dataset_dict = {
            'img_train': X_train, 'features_train': y_train_df,
            'img_test': X_test, 'features_test': y_test_df,
            'config': config
        }

        # Pickle保存
        with open(output_pickle_file, 'wb') as f:
            pickle.dump(dataset_dict, f)

        # --- 設定の更新 ---
        # 1. 読み込むPickleパスを更新 (data_loader用)
        # config['data']['pickle_path'] はプロジェクトルートからの相対パスまたは絶対パス
        # ここでは data_loader.py の仕様 (Path.home() / "ad_project" / config['data']['pickle_path']) に合わせる必要がある
        # 一旦、output_dirからの相対パスを設定する
        rel_path = Path(gen_cfg['output_dir']) / pickle_filename
        config['data']['pickle_path'] = str(rel_path)

        # 2. Run Nameを更新 (engine.pyの出力フォルダ名用)
        # 親フォルダ/cross{i} という構造にする
        # engine.py は output/{run_name} を作成するため、スラッシュを含めることで階層化できる
        base_name = config.get('base_run_name', 'Experiment')
        config['run_name'] = f"{base_name}/cross{fold}"

        # --- 学習実行 ---
        training_phase(config)

def training_phase(config: dict):
    # MLflow設定
    mlflow.set_experiment(config['experiment_name'])
    
    with mlflow.start_run(run_name=config['run_name']):
        mlflow.log_params(config['dataset_generation'])
        mlflow.log_params(config['model'])
        mlflow.log_params(config['training'])
        mlflow.log_param("target_variable", config['data']['target_variable'])
        mlflow.log_param("fold_pickle", config['data']['pickle_path'])
        
        set_seed(config['environment']['seed'])

        # data_loaderから読み込み (指定されたFold用のPickleを読む)
        train_ds, test_ds = get_datasets(config)

        img_shape = next(iter(train_ds))[0]['img_input'].shape
        
        # モデル構築
        model = build_model(input_shape=img_shape, config=config)
        
        # 学習
        run_training(model, train_ds, test_ds, config)

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