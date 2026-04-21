import os
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import concurrent.futures
import multiprocessing

from functions.explainability_utils import (
    load_crop_metadata,
    generate_model_input_reproduction,
    load_exported_model_weights,
    compute_3d_gradcam,
    save_nifti
)
from functions.models import build_model

def process_task(task_args):
    config, sub_id, cross_num = task_args
    try:
        # PyTorch内部のマルチスレッドと競合するのを防ぐため、スレッド数を1に制限
        torch.set_num_threads(2) 
        
        main(config=config, sub_id=sub_id, cross_num=cross_num)
        return f"[SUCCESS] Processed {sub_id} in fold {cross_num}"
    except Exception as e:
        return f"[ERROR] Failed {sub_id} in fold {cross_num}: {e}"

def find_subjects_from_cross_validation(config, cross_num):
    base_dir = Path.home() / "ad_project"
    mapping_path = base_dir / config['paths']['testdata_mapping_path'].format(
        input_region=config['analysis']['input_region'],
        target_metric_name=config['analysis']['target_metric_name'],
        cross_num=cross_num
    )
    
    df = pd.read_excel(mapping_path)
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

    # 1. JSONファイルから共通座標とキャンバスサイズを取得
    json_path = base_dir / "data/processed/crop_metadata.json" 
    crop_ranges, canvas_shape = load_crop_metadata(json_path, input_region)
    
    # 2. 画像生成 (固定座標を使用)
    print("Generating model input using fixed coordinates...")
    base_image_path = base_dir / config['paths']['base_image_path'].format(subject=sub_id)
    mask_image_path = base_dir / config['paths']['mask_image_path'].format(subject=sub_id, input_region=input_region)

    img_input = generate_model_input_reproduction(
        base_path=base_image_path,
        mask_path=mask_image_path,
        crop_ranges=crop_ranges
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
    model_path = os.path.join(model_dir, "model.pth") 
    
    input_shape = (1,) + tuple(canvas_shape)
    
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
    heatmap_save_path = os.path.join(output_dir, "gradcam.nii")
    save_nifti(heatmap, heatmap_save_path)

    mask = img_input > 1e-5
    masked_heatmap = heatmap * mask
    masked_heatmap_save_path = os.path.join(output_dir, "gradcam_masked.nii")
    save_nifti(masked_heatmap, masked_heatmap_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='src_torch/config/analysis_settings.yaml')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel processes')
    args = parser.parse_args()

    print(f"Loading config: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    tasks = []
    for i in range(5):
        sub_ids = find_subjects_from_cross_validation(config, cross_num=i)
        for sub_id in sub_ids:
            tasks.append((config, sub_id, i))

    max_workers = args.workers or max(1, multiprocessing.cpu_count() - 2)
    print(f"Starting parallel processing with {max_workers} processes. Total tasks: {len(tasks)}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_task, tasks)
        for result in results:
            print(result)