import os
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
    
    input_rois = config['analysis']['input_rois']
    target_metrics = config['analysis']['target_metrics']
    score_types = config['analysis']['score_types']
    folds = config['analysis']['cross_validation_folds']
    rois_mapping = config['rois_mapping']
    thresholds = config['analysis']['thresholds']
    
    # マルチプロセスのワーカ数設定
    max_workers = min(10, multiprocessing.cpu_count() - 2)
    
    for input_roi in input_rois:
        sub_rois = rois_mapping.get(input_roi, [])
        if not sub_rois:
            continue
            
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
                        
                    # 当該Foldに存在する被験者IDを取得
                    cross_ids = [d.name for d in cross_dir.iterdir() if d.is_dir()]
                    
                    # マルチプロセス用のタスクリスト作成
                    tasks = []
                    for sub_id in cross_ids:
                        heat_map_path = cross_dir / sub_id / "gradcam_masked.nii"
                        args = (
                            sub_id, heat_map_path, processed_base, 
                            sub_rois, config['paths']['mask_template'], 
                            score_type, thresholds
                        )
                        tasks.append(args)

                    # 並列処理の実行
                    df = pd.DataFrame(columns=sub_rois, index=cross_ids)
                    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                        results = executor.map(process_subject_roi, tasks)
                        for sub_id, scores in results:
                            df.loc[sub_id] = scores
                            
                    df = df.astype(float)
                    importance_dfs.append(df)
                    
                    # Foldごとのグラフ保存
                    mean_dict = df.mean(skipna=True).to_dict()
                    plt.figure(figsize=(8, 8))
                    plt.bar(list(mean_dict.keys()), list(mean_dict.values()), color='skyblue')
                    plt.xticks(rotation=45)
                    plt.grid(axis='y')
                    plt.tight_layout()
                    plt.savefig(analysis_dir / f"{score_prefix}importance_scores_fold_{cross_num+1}.png")
                    plt.close()

                # Excelへ全Foldのデータを書き出し
                if not importance_dfs:
                    continue
                    
                excel_path = analysis_dir / f"{score_prefix}importance_scores.xlsx"
                with pd.ExcelWriter(excel_path) as writer:
                    for cross_num, df in enumerate(importance_dfs):
                        df.to_excel(writer, sheet_name=f'Test_Data_Fold_{cross_num+1}', index=True)

                # 全Foldの平均 (Overall Mean) を計算して追記
                score_mean_df = pd.DataFrame(columns=sub_rois)
                for cross_num, df in enumerate(importance_dfs):
                    mean_dict = df.replace(0, np.nan).mean(skipna=True).to_dict()
                    score_mean_df.loc[f'Fold_{cross_num+1}'] = mean_dict

                overall_mean_dict = score_mean_df.mean(skipna=True).to_dict()
                score_mean_df.loc['Overall_Mean'] = overall_mean_dict
                
                with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                    score_mean_df.to_excel(writer, sheet_name='Overall_Mean', index=True)

                # Overall Mean のグラフ保存
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