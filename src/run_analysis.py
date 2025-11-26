import os
import yaml
import nibabel as nib
import pandas as pd
import numpy as np
import argparse
from functions.explainability_utils import (
    load_exported_model,
    compute_3d_gradcam,
    quantize_with_atlas,
    generate_input_data,
    restore_heatmap_to_original_space
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='analysis_config.yaml')
    args = parser.parse_args()

    print(f"Loading config: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    os.makedirs(config['paths']['output_dir'], exist_ok=True)

    # 1. データの生成 (Base + Mask -> Model Input)
    print("Generating input data from Base and Mask...")
    img_input, transform_info = generate_input_data(
        base_path=config['paths']['base_image_path'],
        mask_path=config['paths']['mask_image_path'],
        canvas_shape=tuple(config['data']['canvas_shape']),
        mode=config['data']['roi_processing_mode']
    )
    
    print(f"Input shape: {img_input.shape}")

    # 2. モデルのロード
    model = load_exported_model(config['paths']['model_dir'])
    
    # 3. Grad-CAMの実行 (Canvas空間でのヒートマップ)
    print("Running Grad-CAM...")
    target_metric = config['analysis']['target_metric_name']
    mode = config['analysis']['score_mode']
    
    heatmap_canvas = compute_3d_gradcam(
        model=model,
        img_array=img_input, # shape: (X, Y, Z) or (X, Y, Z, 1)
        score_mode=mode,
        target_layer=config['model']['target_layer']
    )

    # 4. ヒートマップの座標復元 (Canvas空間 -> Original MNI空間)
    print("Restoring heatmap to original MNI space...")
    heatmap_original = restore_heatmap_to_original_space(heatmap_canvas, transform_info)
    
    # NIfTI保存 (元のAffine情報を使用)
    original_affine = transform_info['affine']
    heatmap_nii = nib.Nifti1Image(heatmap_original, original_affine)
    
    save_name = f"gradcam_{target_metric}_{mode}_restored.nii.gz"
    save_path = os.path.join(config['paths']['output_dir'], save_name)
    nib.save(heatmap_nii, save_path)
    print(f"Restored heatmap saved to: {save_path}")

    # 5. アトラス解析 (復元後の画像を使用するので解剖学的に正しい)
    print("Quantifying with Atlas...")
    roi_scores = quantize_with_atlas(heatmap_nii, atlas_name=config['analysis']['atlas_name'])

    # 結果保存
    df = pd.DataFrame(roi_scores, columns=['ROI_Label', 'Importance_Score'])
    csv_path = os.path.join(config['paths']['output_dir'], f"roi_importance_{target_metric}.csv")
    df.to_csv(csv_path, index=False)
    
    print("\n=== Top 5 Contributing Regions (MNI Space Corrected) ===")
    print(df.head(5))

if __name__ == "__main__":
    main()