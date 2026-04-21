# run_roi_analysis.py
import os
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import multiprocessing
import concurrent.futures

from functions.roi_utils import process_subject_roi

plt.rcParams.update({
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'axes.titlesize': 24
})

def main(config):
    base_dir = Path.home()
    analysis_base = base_dir / config['paths']['analysis_base_dir']
    processed_base = base_dir / config['paths']['processed_data_dir']
    
    # 1. JSONからメタデータを読み込む
    json_path = base_dir / "ad_project/data/processed/crop_metadata.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        crop_metadata = json.load(f)
    
    input_rois = config['analysis']['input_rois']
    target_metrics = config['analysis']['target_metrics']
    score_types = config['analysis']['score_types']
    folds = config['analysis']['cross_validation_folds']
    rois_mapping = config['rois_mapping']
    thresholds = config['analysis']['thresholds']

    max_workers = min(20, multiprocessing.cpu_count() - 2)
    
    for input_roi in input_rois:
        sub_rois = rois_mapping.get(input_roi, [])
        if not sub_rois:
            continue
            
        # JSONから対象となるメインROIの切り出し座標を取得
        if input_roi not in crop_metadata:
            print(f"Error: {input_roi} not found in metadata JSON.")
            continue
            
        ranges_dict = crop_metadata[input_roi]['crop_ranges']
        crop_ranges = [ranges_dict['x'], ranges_dict['y'], ranges_dict['z']]
            
        for target_metric in target_metrics:
            analysis_dir_name = config['paths']['analysis_dir_template'].format(
                input_roi=input_roi, target_metric=target_metric
            )
            analysis_dir = analysis_base / analysis_dir_name
            
            if not analysis_dir.exists():
                print(f"Directory not found, skipping: {analysis_dir}")
                continue

            for score_type in score_types:
                print(f"\nProcessing [{input_roi}] -> [{target_metric}] | Score: {score_type}")
                
                importance_dfs = []
                score_prefix = "" if score_type == "standard" else f"{score_type}_"
                
                for cross_num in range(folds):
                    cross_dir = analysis_dir / f"cross{cross_num}"
                    if not cross_dir.exists():
                        continue
                        
                    cross_ids = [d.name for d in cross_dir.iterdir() if d.is_dir()]
                    
                    tasks = []
                    for sub_id in cross_ids:
                        out_subject_dir = cross_dir / sub_id
                        heat_map_path = out_subject_dir / "gradcam_masked.nii"
                        
                        args = (
                            sub_id, heat_map_path, processed_base, 
                            sub_rois, config['paths']['mask_template'], 
                            score_type, thresholds, crop_ranges,
                            out_subject_dir # ★ 追加: 保存先のディレクトリを渡す
                        )
                        tasks.append(args)

                    df = pd.DataFrame(columns=sub_rois, index=cross_ids)

                    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                        results = executor.map(process_subject_roi, tasks)
                        for sub_id, scores, _ in results: 
                            df.loc[sub_id] = scores

                    df = df.astype(float)
                    importance_dfs.append(df)
                    
                    mean_dict = df.mean(skipna=True).to_dict()
                    plt.figure(figsize=(8, 8))
                    plt.bar(list(mean_dict.keys()), list(mean_dict.values()), color='skyblue')
                    plt.xticks(rotation=45)
                    plt.grid(axis='y')
                    plt.tight_layout()
                    plt.savefig(analysis_dir / f"{score_prefix}importance_scores_fold_{cross_num+1}.png")
                    plt.close()

                if not importance_dfs:
                    continue
                    
                excel_path = analysis_dir / f"{score_prefix}importance_scores.xlsx"
                with pd.ExcelWriter(excel_path) as writer:
                    for cross_num, df in enumerate(importance_dfs):
                        df.to_excel(writer, sheet_name=f'Test_Data_Fold_{cross_num+1}', index=True)

                score_mean_df = pd.DataFrame(columns=sub_rois)
                for cross_num, df in enumerate(importance_dfs):
                    mean_dict = df.replace(0, np.nan).mean(skipna=True).to_dict()
                    score_mean_df.loc[f'Fold_{cross_num+1}'] = mean_dict

                overall_mean_dict = score_mean_df.mean(skipna=True).to_dict()
                score_mean_df.loc['Overall_Mean'] = overall_mean_dict
                
                with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                    score_mean_df.to_excel(writer, sheet_name='Overall_Mean', index=True)

                plt.figure(figsize=(8, 8))
                plt.bar(list(overall_mean_dict.keys()), list(overall_mean_dict.values()), color='skyblue')
                plt.xticks(rotation=45)
                plt.grid(axis='y')
                plt.tight_layout()
                plt.savefig(analysis_dir / f"{score_prefix}importance_scores_mean.png")
                plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='roi_importance_config.yaml')
    args = parser.parse_args()

    print(f"Loading config: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    main(config)