import os
import pandas as pd
import numpy as np
import nibabel as nib
import math
from tqdm import tqdm

from .image_preprocessing import crop_roi, pad_to_canvas, normalize_intensity

def load_and_match_data(data_dir, csv_path, path_templates, target_columns):
    """
    臨床データと画像ファイルを読み込み, IDを基準に紐づける
    """
    try:
        df_clinical = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"エラー: 臨床データファイルが見つかりません: {csv_path}")
        return pd.DataFrame()

    required_columns = ['subject_id'] + target_columns

    if not all(col in df_clinical.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df_clinical.columns]
        print(f"エラー: CSVファイルにカラムが見つかりません: {', '.join(missing_cols)}")
        return pd.DataFrame()
    
    # 異なるシート間でsubject_idが重複している場合の警告と処理
    if df_clinical['subject_id'].duplicated().any():
        num_duplicates = df_clinical['subject_id'].duplicated().sum()
        print(f"警告: 異なるシート間で {num_duplicates} 件の 'subject_id' の重複が見つかりました. 最初の出現データを使用します. ")
        df_clinical = df_clinical.drop_duplicates(subset=['subject_id'], keep='first')

    # 必要なカラムのみ抽出し、欠損値を持つ行を削除
    df_clinical = df_clinical[required_columns].dropna()

    file_data = []
    print("臨床データと画像ファイルのマッチングを開始します...")
    for _, row in df_clinical.iterrows():
        sub_id = row['subject_id']
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

def create_dataset(df, mode, target_columns, canvas_shape=None):
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
        features.append(row[target_columns])

    img_array = np.array(images)[..., np.newaxis]
    features_df = pd.DataFrame(features).reset_index(drop=True)

    return img_array, features_df