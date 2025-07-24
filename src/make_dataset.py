import os
import pandas as pd
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
import pickle
import math
from tqdm import tqdm

#--- 1. ヘルパー関数 ---
def crop_roi(roi_array):
    if np.all(roi_array == 0):
        return np.zeros((1, 1, 1), dtype=roi_array.dtype)
    indeces = np.array(np.where(roi_array != 0))
    min_coords = indeces.min(axis=1)
    max_coords = indeces.max(axis=1)
    return roi_array[min_coords[0]:max_coords[0]+1,
                     min_coords[1]:max_coords[1]+1,
                     min_coords[2]:max_coords[2]+1]

def pad_to_canvas(cropped_array, canvas_shape):
    canvas = np.zeros(canvas_shape, dtype=cropped_array.dtype)
    start_coords = [(cs - da) // 2 for cs, da in zip(canvas_shape, cropped_array.shape)]
    end_coords = [sc + da for sc, da in zip(start_coords, cropped_array.shape)]
    canvas[start_coords[0]:end_coords[0],
              start_coords[1]:end_coords[1],
              start_coords[2]:end_coords[2]] = cropped_array
    return canvas

def normalize_intensity(array):
    min_val, max_val = array.min(), array.max()
    if max_val - min_val > 0:
        return (array - min_val) / (max_val - min_val)
    return array

#--- 2. 設定セクション ---
#ベースディレクトリ
BASE_DIR = '/home/matsuda/ad_project'
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data/derivatives/processed_data')
EXCEL_PATH = os.path.join(BASE_DIR, 'data/raw_data/■■MRI-ASL症例リスト入力書式2024年10月21日作成版（山大版）20241212最終（匿名化）.xlsx')

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
EXCEL_COLUMNS = ['MMSE', 'CDR', 'HDSR']

# データ分割の割合
TEST_SIZE = 0.2

# 出力ファイル名
OUTPUT_PICKLE_FILE = os.path.join(BASE_DIR, f'data/processed/dataset_{MASK_NAME}.pkl')

# --- 3. データ処理関数 ---
def load_and_match_data(data_dir, excel_path, path_templates, excel_columns):
    try:
        all_sheets_dict = pd.read_excel(excel_path, sheet_name=None, dtype={'研究用匿名化ID': str})
    except FileNotFoundError:
        print(f"エラー: 臨床データファイルが見つかりません: {excel_path}")
        return pd.DataFrame()

    if not all_sheets_dict:
        print(f"エラー: Excelファイル '{os.path.basename(excel_path)}' にシートが見つかりません. ")
        return pd.DataFrame()

    df_clinical = pd.concat(all_sheets_dict.values(), ignore_index=True)
    print(f"Excelの全シートから合計 {len(df_clinical)} 件の臨床データを読み込みました. ")

    required_columns = ['研究用匿名化ID'] + excel_columns

    if not all(col in df_clinical.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df_clinical.columns]
        print(f"エラー: 結合後のデータにカラムが見つかりません: {', '.join(missing_cols)}")
        return pd.DataFrame()
    
    # 異なるシート間でsubject_idが重複している場合の警告と処理
    if df_clinical['研究用匿名化ID'].duplicated().any():
        num_duplicates = df_clinical['研究用匿名化ID'].duplicated().sum()
        print(f"警告: 異なるシート間で {num_duplicates} 件の '研究用匿名化ID' の重複が見つかりました. 最初の出現データを使用します. ")
        df_clinical = df_clinical.drop_duplicates(subset=['研究用匿名化ID'], keep='first')

    # 必要なカラムのみ抽出し、欠損値を持つ行を削除
    df_clinical = df_clinical[required_columns].dropna()

    file_data = []
    print("臨床データと画像ファイルのマッチングを開始します...")
    for _, row in df_clinical.iterrows():
        sub_id = row['研究用匿名化ID']
        paths = {key: os.path.join(data_dir, sub_id, template.format(subject=sub_id))
                 for key, template in path_templates.items()}
        
        if all(os.path.exists(p) for p in paths.values()):
            data_row = row.to_dict()
            data_row.update(paths)
            file_data.append(data_row)
        else:
            missing_files = [os.path.basename(p) for p in paths.values() if not os.path.exists(p)]
            print(f"警告: 被験者 {sub_id} のファイルが見つからないためスキップします。 (不足ファイル: {', '.join(missing_files)})")

    return pd.DataFrame(file_data)

def determine_target_canvas_size(df):
    cropped_shapes = []
    print("最適なキャンバスサイズを計算中...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        base_img = nib.load(row['base']).get_fdata()
        mask_img = nib.load(row['mask']).get_fdata()

        roi_array = base_img * (mask_img > 0.5)
        cropped_roi = crop_roi(roi_array)
        cropped_shapes.append(cropped_roi.shape)
    
    max_dims = np.max(cropped_shapes, axis=0)
    canvas_shape = tuple(math.ceil(d / 16) * 16 for d in max_dims)
    return canvas_shape

def create_dataset(df, mode, excel_columns, canvas_shape=None):
    images, features = [], []
    print(f"データセットを生成中 (モード: {mode})...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        base_img = nib.load(row['base']).get_fdata().astype(np.float32)
        mask_img = nib.load(row['mask']).get_fdata().astype(np.float32)
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
        features.append(row[excel_columns])

    img = np.array(images)[..., np.newaxis]
    features = pd.DataFrame(features).reset_index(drop=True)

    return img, features

#--- 4. メイン処理 ---
if __name__ == '__main__':
    print("--- データセット作成開始 ---")

    # 1. データ情報の読み込みとマッチング
    matched_df = load_and_match_data(PROCESSED_DATA_DIR, EXCEL_PATH, PATH_TEMPLATES, EXCEL_COLUMNS)
    if matched_df.empty:
        print("処理対象データが見つかりませんでした. パスやファイル名を確認してください. ")
    else:
        print(f"\nマッチング成功: {len(matched_df)}名のデータで処理を進めます. ")

        # 2. (crop_and_padモード時) 最適なキャンバスサイズを決定
        target_canvas_size = None
        if ROI_PROCESSING_MODE == 'crop_and_pad':
            target_canvas_size = determine_target_canvas_size(matched_df)
            print(f"\n決定したキャンバスサイズ: {target_canvas_size}")
        
        # 3. 患者単位でIDを訓練用とテスト用に分割
        stratify_column = EXCEL_COLUMNS[0]
        train_df, test_df = train_test_split(matched_df, test_size=TEST_SIZE, random_state=42, stratify=pd.cut(matched_df[stratify_column], bins=5, labels=False, duplicates='drop'))

        print("\nデータ分割:")
        print(f"訓練用データ: {len(train_df)}名")
        print(f"テスト用データ: {len(test_df)}名")

        # 4. 各データセットの画像とラベルを生成
        img_train, features_train = create_dataset(train_df, ROI_PROCESSING_MODE, EXCEL_COLUMNS, target_canvas_size)
        img_test, features_test = create_dataset(test_df, ROI_PROCESSING_MODE, EXCEL_COLUMNS, target_canvas_size)

        # 5. Pickle形式で保存
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