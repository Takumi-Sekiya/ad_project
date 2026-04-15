import os
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch

from functions.explainability_utils import (
    get_canvas_shape_from_pickle,
    generate_model_input_reproduction,
    load_exported_model_weights,
    compute_3d_gradcam,
    save_nifti
)
from functions.models import build_model

def find_subjects(raw_data_dir, prefixes):
    subject_ids = set()
    for prefix in prefixes:
        for path in raw_data_dir.glob(prefix):
            if path.is_dir():
                subject_ids.add(path.name)
    
    subject_ids.discard(".")
    subject_ids.discard("..")

    subject_ids = sorted(list(subject_ids))

    # 特定の被験者を除外 (元のロジック維持)
    for remove_id in ["AMC_SIK_0019_77_0_1", "AMC_SIK_0021_89_0_1", "AMC_SIK_0044_78_1_1", "AMC_SIK_0044_79_1_3"]:
        if remove_id in subject_ids:
            subject_ids.remove(remove_id)
            
    return subject_ids

def find_subjects_from_cross_validation(config, cross_num):
    base_dir = Path.home() / "ad_project"
    mapping_path = base_dir / config['paths']['testdata_mapping_path'].format(
        target_metric_name=config['analysis']['target_metric_name']
    )
    
    df = pd.read_excel(mapping_path, sheet_name=f"Test_Data_Fold_{cross_num+1}")
    subject_ids = df['subject_id'].tolist()
    
    return subject_ids

def main(config, sub_id, cross_num=0):
    input_region = config['analysis']['input_region']
    target_metric_name = config['analysis']['target_metric_name']

    base_dir = Path.home() / "ad_project"

    output_dir = base_dir / config['paths']['output_dir'].format(
        input_region=input_region,
        target_metric_name=target_metric_name,
        sub_id=sub_id,
        cross_num=cross_num 
    )
    os.makedirs(output_dir, exist_ok=True)

    # 1. キャンバスサイズ取得
    pickle_path = base_dir / config['paths']['dataset_pickle_path'].format(
        input_region=input_region,
        target_metric_name=target_metric_name
    )
    canvas_shape = get_canvas_shape_from_pickle(pickle_path)
    
    # 2. 画像生成
    print("Generating model input using reproduced preprocessing logic...")
    base_image_path = base_dir / config['paths']['base_image_path'].format(subject=sub_id)
    mask_image_path = base_dir / config['paths']['mask_image_path'].format(subject=sub_id, input_region=input_region)

    img_input = generate_model_input_reproduction(
        base_path=base_image_path,
        mask_path=mask_image_path,
        canvas_shape=canvas_shape
    )
    print(f"Generated input image shape: {img_input.shape}")

    input_save_path = os.path.join(output_dir, "input_roi_image.nii")
    save_nifti(img_input, input_save_path)

    # 3. モデル構築 & ロード
    print("Building model architecture and loading weights...")
    model_dir = base_dir / config['paths']['model_dir'].format(
        input_region=input_region,
        target_metric_name=target_metric_name,
        cross_num=cross_num
    )
    # 拡張子を.kerasから.pthに変更したことを想定
    model_path = os.path.join(model_dir, "model.pth") 
    
    # 画像の入力形状 (in_channels=1 を先頭にするPyTorch仕様)
    input_shape = (1,) + canvas_shape
    
    # base_configを読み込んでモデルをインスタンス化
    base_config_path = base_dir / 'src' / 'config' / 'config.yaml'
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    model_instance = build_model(input_shape, base_config)
    model = load_exported_model_weights(model_instance, model_path)

    # 4. Grad-CAM実行
    mode = config['analysis']['score_mode']
    print(f"Running Grad-CAM for {target_metric_name} ({mode})...")
    
    target_layer_name = config['model'].get('target_layer', None)
    
    heatmap = compute_3d_gradcam(
        model=model,
        img_array=img_input,
        score_mode=mode,
        target_layer_name=target_layer_name
    )

    # 5. ヒートマップ保存
    heatmap_save_path = os.path.join(output_dir, f"gradcam.nii")
    save_nifti(heatmap, heatmap_save_path)

    mask = img_input > 1e-5
    masked_heatmap = heatmap * mask
    masked_heatmap_save_path = os.path.join(output_dir, f"gradcam_masked.nii")
    save_nifti(masked_heatmap, masked_heatmap_save_path)

    print("\n=== プロトタイプ作成完了 ===")
    print(f"処理モード: {mode} (increase=正の寄与, decrease=負の寄与)")
    print(f"1. 入力画像 (Background): {input_save_path}")
    print(f"2. ヒートマップ (Overlay): {heatmap_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='src/config/analysis_settings.yaml')
    args = parser.parse_args()

    print(f"Loading config: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for i in range(5):
        sub_ids = find_subjects_from_cross_validation(config, cross_num=i)
        for sub_id in sub_ids:
            try:
                main(config=config, sub_id=sub_id, cross_num=i)
            except Exception as e:
                print(f"Error processing {sub_id} in fold {i}: {e}")
                continue