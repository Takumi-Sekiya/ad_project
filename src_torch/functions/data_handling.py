import os
import json
import pandas as pd
import numpy as np
import nibabel as nib
import math
from tqdm import tqdm

from .image_preprocessing import get_non_zero_bounds, crop_by_ranges, normalize_intensity

def load_and_preprocess_mask(mask_path, threshold=100):
    """
    マスク画像を読み込み、必要に応じて2値化(0 or 1)するヘルパー関数
    """
    mask_nii = nib.load(mask_path)
    mask_data = mask_nii.get_fdata().astype(np.float32)

    if mask_data.max() > 1.0:
        mask_data = np.where(mask_data >= threshold, 1.0, 0.0)
    
    return mask_data

def load_and_match_data(data_dir, csv_path, path_templates, target_columns, allowed_diagnoses=None):
    """
    臨床データと画像ファイルを読み込み, IDを基準に紐づける
    """
    try:
        df_clinical = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"エラー: 臨床データファイルが見つかりません: {csv_path}")
        return pd.DataFrame()

    # 'diagnosis' を必須カラムチェックに含める
    check_columns = ['subject_id'] + target_columns
    if allowed_diagnoses is not None and 'diagnosis' not in check_columns:
        check_columns.append('diagnosis')

    if not all(col in df_clinical.columns for col in check_columns):
        missing_cols = [col for col in check_columns if col not in df_clinical.columns]
        print(f"エラー: CSVファイルにカラムが見つかりません: {', '.join(missing_cols)}")
        return pd.DataFrame()
    
    # 異なるシート間でsubject_idが重複している場合の警告と処理
    if df_clinical['subject_id'].duplicated().any():
        num_duplicates = df_clinical['subject_id'].duplicated().sum()
        print(f"警告: 異なるシート間で {num_duplicates} 件の 'subject_id' の重複が見つかりました. 最初の出現データを使用します. ")
        df_clinical = df_clinical.drop_duplicates(subset=['subject_id'], keep='first')

    # 診断名のフィルタリングを実施
    if allowed_diagnoses is not None:
        df_clinical = df_clinical.dropna(subset=['diagnosis']) # 診断名が欠損している行は除外
        
        # allowed_diagnosesのいずれかの文字列が含まれているかチェック (正規表現のOR条件を作成)
        pattern = '|'.join(allowed_diagnoses)
        initial_count = len(df_clinical)
        
        # 部分一致でフィルタリング
        df_clinical = df_clinical[df_clinical['diagnosis'].astype(str).str.contains(pattern, na=False, regex=True)]
        
        filtered_count = len(df_clinical)
        print(f"診断名フィルタを適用: {initial_count}件 -> {filtered_count}件に絞り込みました。 (条件: {allowed_diagnoses})")

    # 必要なカラムのみ抽出処理に使うために欠損値を持つ行を削除
    required_columns = ['subject_id'] + target_columns
    df_clinical = df_clinical.dropna(subset=required_columns)

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
            print(f"{sub_id} skipped.", end=' ')

    print("\nマッチング完了.")
    return pd.DataFrame(file_data)

def determine_global_crop_ranges(df, target_multiple=16):
    """
    全画像のマスクから共通のROIバウンディングボックスを計算し、
    指定の倍数(16または8)のサイズになるように境界を拡張する。
    """
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max_incl = np.array([-np.inf, -np.inf, -np.inf])
    img_shape = None

    print("全画像から共通のROI座標（Global Bounding Box）を計算中...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        mask_img = load_and_preprocess_mask(row['mask'], threshold=100)
        if img_shape is None:
            img_shape = mask_img.shape # 元画像のサイズを取得（はみ出し防止用）

        min_c, max_c = get_non_zero_bounds(mask_img)
        if min_c is not None:
            global_min = np.minimum(global_min, min_c)
            global_max_incl = np.maximum(global_max_incl, max_c)
    
    global_min = global_min.astype(int)
    # スライシングのために最大値は +1 (排他的) にしておく
    global_max_excl = global_max_incl.astype(int) + 1 

    current_sizes = global_max_excl - global_min
    # target_multipleの倍数に切り上げ
    target_sizes = np.ceil(current_sizes / target_multiple).astype(int) * target_multiple

    diffs = target_sizes - current_sizes
    pad_before = diffs // 2
    pad_after = diffs - pad_before

    new_min = global_min - pad_before
    new_max_excl = global_max_excl + pad_after

    # 元画像の端を超えないように補正するロジック
    for i in range(3):
        if new_min[i] < 0:
            # 左にはみ出した分を右に回す
            new_max_excl[i] += abs(new_min[i])
            new_min[i] = 0
        if new_max_excl[i] > img_shape[i]:
            # 右にはみ出した分を左に回す
            new_min[i] -= (new_max_excl[i] - img_shape[i])
            new_max_excl[i] = img_shape[i]
            
        # それでもはみ出る（ROIが画像サイズより大きい等）場合はゼロで防ぐ
        new_min[i] = max(0, new_min[i])

    ranges = [[int(new_min[i]), int(new_max_excl[i])] for i in range(3)]
    target_shape = tuple(int(x) for x in target_sizes)
    
    return ranges, target_shape

def save_crop_metadata(json_path, roi_name, ranges, target_shape):
    """
    算出した座標とサイズをJSONファイルに追記（既存データ保持）で保存する。
    """
    data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            pass # ファイルが空、または壊れている場合は空の辞書からスタート

    data[roi_name] = {
        "crop_ranges": {
            "x": ranges[0],
            "y": ranges[1],
            "z": ranges[2]
        },
        "target_shape": target_shape
    }

    # 親ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"メタデータをJSONに保存しました: {json_path}")

def create_dataset(df, mode, target_columns, crop_ranges=None):
    images, features = [], []
    print(f"データセットを生成中 (モード: {mode})...")
    cols_to_extract = target_columns.copy()
    if 'subject_id' not in cols_to_extract:
        cols_to_extract.insert(0, 'subject_id')

    for _, row in tqdm(df.iterrows(), total=len(df)):
        base_img = nib.load(row['base']).get_fdata().astype(np.float32)
        mask_img = load_and_preprocess_mask(row['mask'], threshold=50)
        roi_array = base_img * (mask_img > 0.5)

        if mode == 'crop_and_pad':
            if crop_ranges is None:
                raise ValueError("crop_and_padモードにはcrop_rangesが必要です")
            
            # 従来のような「各画像のクロップ＆中央パディング」ではなく、
            # 指定された「共通の絶対座標」での切り出しを行うだけでサイズが揃う
            processed_img = crop_by_ranges(roi_array, crop_ranges)
            processed_img = normalize_intensity(processed_img)
            
        elif mode == 'simple_mask':
            processed_img = normalize_intensity(roi_array)
        else:
            raise ValueError(f"未知のROI処理モードです: {mode}")
        
        images.append(processed_img)
        features.append(row[cols_to_extract])

    img_array = np.array(images)[..., np.newaxis]
    features_df = pd.DataFrame(features).reset_index(drop=True)

    return img_array, features_df