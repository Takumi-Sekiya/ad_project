import os
import yaml
import argparse
import numpy as np
from pathlib import Path

from functions.explainability_utils import (
    get_canvas_shape_from_pickle,
    generate_model_input_reproduction,
    load_exported_model,
    compute_3d_gradcam,
    save_nifti
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='analysis_settings.yaml')
    args = parser.parse_args()

    # 設定ロード
    print(f"Loading config: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # パス情報の展開 (チルダ展開などに対応するためPathを使用)
    base_dir = Path.home() / "ad_project"

    # 出力ディレクトリ作成
    output_dir = base_dir / config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # 1. Pickleファイルからキャンバスサイズを動的に取得
    pickle_path = base_dir / config['paths']['dataset_pickle_path']
    canvas_shape = get_canvas_shape_from_pickle(pickle_path)
    
    # 2. 画像生成 (data_handling.pyのロジックを完全再現)
    print("Generating model input using reproduced preprocessing logic...")

    sub_id = config['paths']['sub_id']
    base_image_path = base_dir / config['paths']['base_image_path'].format(subject=sub_id)
    mask_image_path = base_dir / config['paths']['mask_image_path'].format(subject=sub_id)

    img_input = generate_model_input_reproduction(
        base_path=base_image_path,
        mask_path=mask_image_path,
        canvas_shape=canvas_shape
    )
    print(f"Generated input image shape: {img_input.shape}")

    # 確認用: 入力画像の保存
    input_save_path = os.path.join(output_dir, "input_roi_image.nii")
    save_nifti(img_input, input_save_path)

    # 3. モデルロード
    print("Loading model...")
    model_dir = base_dir / config['paths']['model_dir']
    model = load_exported_model(model_dir)

    # 4. Grad-CAM実行
    target_metric = config['analysis']['target_metric_name']
    mode = config['analysis']['score_mode']
    print(f"Running Grad-CAM for {target_metric} ({mode})...")
    
    heatmap = compute_3d_gradcam(
        model=model,
        img_array=img_input,
        score_mode=mode,
        target_layer=config['model']['target_layer']
    )

    # 5. ヒートマップ保存
    heatmap_save_path = os.path.join(output_dir, f"gradcam_{target_metric}.nii")
    save_nifti(heatmap, heatmap_save_path)

    print("\n=== プロトタイプ作成完了 ===")
    print(f"処理モード: {mode} (increase=正の寄与, decrease=負の寄与)")
    print("以下のファイルをビューワーで重ねて確認してください:")
    print(f"1. 入力画像 (Background): {input_save_path}")
    print(f"2. ヒートマップ (Overlay): {heatmap_save_path}")

if __name__ == "__main__":
    main()