import yaml
import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from functions.data_handling import load_and_match_data, determine_target_canvas_size, create_dataset



#ベースディレクトリ
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'derivatives' / 'processed_data'
CSV_PATH = BASE_DIR / 'data' / 'processed' / 'structured_data.csv'

#--- 実験に合わせて変更するパラメータ ---
# 使用する画像のパスを定義するテンプレート
# {subject} の部分が被験者IDに置換される
PATH_TEMPLATES = {
    'base': 'spm/norm/mw{subject}_T1w.nii',      # 元画像
    #'mask': 'spm/norm/w{subject}_mask-hippocampus.nii', # マスク (例: 海馬)
    #'mask': 'spm/norm/wc1{subject}_T1w.nii', # 灰白質をマスクとして使う場合など
    #'mask': 'spm/norm/w{subject}_mask-prefrontal-cortex.nii',
    #'mask': 'spm/norm/w{subject}_mask-parietal-lobe.nii',
    #'mask': 'spm/norm/w{subject}_mask-occipital-lobe.nii',
    #'mask': 'spm/norm/w{subject}_mask-temporal-lobe.nii',
    'mask': 'spm/norm/w{subject}_mask-brain-stem.nii',
}
MASK_NAME = 'brain-stem'

# ROI抽出の方法: 
# 'crop_and_pad': 周囲の不要な0領域を削除し、全データで統一したサイズに整形
# 'simple_mask': 単純にマスクを乗算するだけ（全脳画像サイズになる）
ROI_PROCESSING_MODE = 'crop_and_pad' 

# 予測したい指標（Excelファイルのカラム名）
EXCEL_COLUMNS = ['MMSE', 'CDR', 'gm_atrophy', 'severity']

# データ分割の割合
TEST_SIZE = 0.2

# 出力ファイル名
OUTPUT_PICKLE_FILE = BASE_DIR / 'data' / 'processed' / f'dataset_{MASK_NAME}.pkl'




def main():
    print("--- データセット作成開始 ---")

    # 1. 設定の読み込み
    BASE_DIR = Path(__file__).resolve().parent.parent
    with open(BASE_DIR / 'src' / 'config' / 'make_dataset_settings.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 設定値を変数に変換
    data_cfg = config['data']
    dataset_cfg = config['dataset']
    training_cfg = config['training']
    output_cfg = config['output']

    # パスを絶対パスに変換
    processed_data_dir = BASE_DIR / data_cfg['processed_dir']
    csv_path = BASE_DIR / data_cfg['csv_path']
    output_dir = BASE_DIR/ data_cfg['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pickle_file = output_dir / output_cfg['filename_template'].format(mask_name=dataset_cfg['mask_name'])

    # 2. データ情報の読み込みとマッチング
    matched_df = load_and_match_data(processed_data_dir, csv_path, dataset_cfg['path_templates'], dataset_cfg['target_columns'])
    if matched_df.empty:
        print("処理対象データが見つかりませんでした. 処理を中断します. ")
        return
    print(f"\nマッチング成功: {len(matched_df)}名のデータで処理を進めます. ")

    # 3. キャンバスサイズを決定
    target_canvas_size = None
    if ROI_PROCESSING_MODE == 'crop_and_pad':
        target_canvas_size = determine_target_canvas_size(matched_df)
        print(f"\n決定したキャンバスサイズ: {target_canvas_size}")
        
    # 4. 患者単位でIDを訓練用とテスト用に分割
    stratify_column = dataset_cfg['target_columns'][0]
    bins = pd.cut(matched_df[stratify_column], bins=5, labels=False, duplicates='drop')
    train_df, test_df = train_test_split(
        matched_df,
        test_size=training_cfg['test_size'],
        random_state=training_cfg['random_state'],
        stratify=bins
    )

    print("\nデータ分割:")
    print(f"訓練用データ: {len(train_df)}名")
    print(f"テスト用データ: {len(test_df)}名")

    # 5. 各データセットの画像とラベルを生成
    img_train, features_train = create_dataset(
        train_df,
        dataset_cfg['roi_processing_mode'],
        dataset_cfg['target_columns'],
        target_canvas_size
    )
    img_test, features_test = create_dataset(
        test_df,
        dataset_cfg['roi_processing_mode'],
        dataset_cfg['target_columns'],
        target_canvas_size
    )

    # 6. Pickle形式で保存
    dataset_dict = {
        'img_train': img_train, 'features_train': features_train,
        'img_test': img_test, 'features_test': features_test
    }
    print(f"\nデータセットをpickleファイルに保存中: {OUTPUT_PICKLE_FILE}")
    with open(OUTPUT_PICKLE_FILE, 'wb') as f:
        pickle.dump(dataset_dict, f)

    print("\n--- データセット作成完了 ---")
    print(f"使用したベース画像テンプレート: {PATH_TEMPLATES['base']}")
    print(f"使用したマスクテンプレート: {PATH_TEMPLATES['mask']}")
    print(f"抽出特徴量: {EXCEL_COLUMNS}")
    print(f"ROI処理モード: {ROI_PROCESSING_MODE}")
        
    print("\n生成されたデータセットの形状と型:")
    print(f"img_train shape: {img_train.shape}, type: {type(img_train)}")
    print(f"features_train shape: {features_train.shape}, type: {type(features_train)}")
    print("features_train の内容 (先頭5行):")
    print(features_train.head())
    print("-" * 20)
    print(f"img_test shape: {img_test.shape}, type: {type(img_test)}")
    print(f"features_test shape: {features_test.shape}, type: {type(features_test)}")

if __name__ == '__main__':
    main()