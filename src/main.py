import argparse
import yaml
import pickle
import pandas as pd
from pathlib import Path
import mlflow
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 既存モジュールのインポート
from functions.data_handling import load_and_match_data, determine_target_canvas_size, create_dataset
from functions.utils import set_seed
from functions.data_loader import get_datasets
from functions.models import build_model
from functions.engine import run_training

def override_config(config: dict, roi: str, target: str):
    """
    コマンドライン引数で指定されたROIとターゲットに合わせてConfigを上書きする
    """
    if roi:
        print(f">> Config Override: Mask ROI -> {roi}")
        # 1. マスク名の上書き
        config['dataset_generation']['mask_name'] = roi
        
        # 2. マスクファイルパスのテンプレート更新
        # 前提: マスクファイル名は '...mask-{roi}.nii' という規則に従う
        base_mask_template = config['dataset_generation']['path_templates']['mask']
        # 既存のテンプレートが具体名(prefrontal-cortex等)を含んでいる場合に対応するため、
        # ディレクトリ構造とプレフィックスを維持しつつ、後半を差し替えるロジックにするか、
        # シンプルにフォーマット文字列として再定義します。
        # ここではYAMLの設定を信じて {roi} 部分があるか、もしくは決め打ちで構築します。
        # 安全策として、mask-{roi}.nii の形式を強制します。
        config['dataset_generation']['path_templates']['mask'] = f'spm/norm/w{{subject}}_mask-{roi}.nii'

    if target:
        print(f">> Config Override: Target -> {target}")
        # 3. ターゲット変数の更新
        config['data']['target_variable'] = target
        
        # 4. 抽出カラムの限定（欠損値対策：ターゲットのみにする）
        config['dataset_generation']['columns_to_extract'] = [target]

    # 5. Run Nameの更新 (MLflowでの識別用)
    current_roi = config['dataset_generation']['mask_name']
    current_target = config['data']['target_variable']
    config['run_name'] = f"{config['model']['name']}_{current_roi}_to_{current_target}"
    
    return config

def generate_dataset_phase(config: dict, base_dir: Path) -> Path:
    print("\n=== Phase 1: データセット生成を開始 ===")
    
    gen_cfg = config['dataset_generation']
    training_cfg = config['training']
    
    processed_data_dir = base_dir / gen_cfg['raw_data_base_dir']
    csv_path = base_dir / gen_cfg['clinical_csv_path']
    output_dir = base_dir / gen_cfg['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ファイル名生成
    pickle_filename = gen_cfg['filename_template'].format(
        mask_name=gen_cfg['mask_name'],
        target_name=config['data']['target_variable']
    )
    output_pickle_file = output_dir / pickle_filename

    # マッチング実行
    # ここで columns_to_extract は override_config により [target] のみになっているはずです
    matched_df = load_and_match_data(
        processed_data_dir, 
        csv_path, 
        gen_cfg['path_templates'], 
        gen_cfg['columns_to_extract']
    )
    
    if matched_df.empty:
        raise ValueError("処理対象データが見つかりませんでした。")
    print(f"マッチング成功: {len(matched_df)}名のデータを使用します。")

    # キャンバスサイズ決定
    target_canvas_size = None
    if gen_cfg['roi_processing_mode'] == 'crop_and_pad':
        target_canvas_size = determine_target_canvas_size(matched_df)
        print(f"決定したキャンバスサイズ: {target_canvas_size}")

    # データ分割
    stratify_target = config['data']['target_variable']
    # ターゲットに欠損がある行は load_and_match_data ですでに落ちているはずですが念のため
    matched_df = matched_df.dropna(subset=[stratify_target])
    
    #bins = pd.cut(matched_df[stratify_target], bins=5, labels=False, duplicates='drop')
    bins = pd.qcut(matched_df[stratify_target], q=5, labels=False, duplicates="drop")
    train_df, test_df = train_test_split(
        matched_df,
        test_size=training_cfg['test_size'],
        random_state=training_cfg['random_state'],
        stratify=bins
    )
    print(f"データ分割完了 - Train: {len(train_df)}, Test: {len(test_df)}")

    # 画像生成
    print("Trainデータセット生成中...")
    img_train, features_train = create_dataset(
        train_df,
        gen_cfg['roi_processing_mode'],
        gen_cfg['columns_to_extract'],
        target_canvas_size
    )
    print("Testデータセット生成中...")
    img_test, features_test = create_dataset(
        test_df,
        gen_cfg['roi_processing_mode'],
        gen_cfg['columns_to_extract'],
        target_canvas_size
    )

    # 保存
    dataset_dict = {
        'img_train': img_train, 'features_train': features_train,
        'img_test': img_test, 'features_test': features_test,
        'config': config
    }
    
    print(f"データセットを保存中: {output_pickle_file}")
    with open(output_pickle_file, 'wb') as f:
        pickle.dump(dataset_dict, f)
        
    return output_pickle_file

def training_phase(config: dict):
    print("\n=== Phase 2: モデル学習と評価を開始 ===")
    
    mlflow.set_experiment(config['experiment_name'])
    
    # Run Name を設定（override_configで更新されたものを使用）
    with mlflow.start_run(run_name=config['run_name']):
        mlflow.log_params(config['dataset_generation'])
        mlflow.log_params(config['model'])
        mlflow.log_params(config['training'])
        mlflow.log_param("target_variable", config['data']['target_variable'])
        
        set_seed(config['environment']['seed'])

        train_ds, test_ds = get_datasets(config)

        img_shape = next(iter(train_ds))[0]['img_input'].shape
        print(f"入力画像形状: {img_shape}")
        
        model = build_model(input_shape=img_shape, config=config)
        
        run_training(model, train_ds, test_ds, config)

def main(args):
    base_dir = Path(__file__).resolve().parent.parent
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Configを動的に書き換える
    config = override_config(config, args.roi, args.target)

    try:
        # Phase 1: データセット作成
        generated_pickle_path = generate_dataset_phase(config, base_dir)
        
        # Phase 2: 学習へのパス引き渡し
        rel_path = Path(config['dataset_generation']['output_dir']) / generated_pickle_path.name
        config['data']['pickle_path'] = str(rel_path)
        
        # Phase 2: 学習実行
        training_phase(config)

    except Exception as e:
        print(f"\nエラーが発生しました ({config['run_name']}): {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='src/config/config.yaml', help='Path to the base config file')
    # 追加引数: ROIとターゲット
    parser.add_argument('--roi', type=str, default=None, help='Override ROI name (e.g. hippocampus)')
    parser.add_argument('--target', type=str, default=None, help='Override target variable (e.g. MMSE)')
    
    args = parser.parse_args()
    main(args)