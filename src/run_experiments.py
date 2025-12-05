# python src/run_experiments.py

import subprocess
import sys
import time

def run_experiments():
    # --- 設定エリア ---
    
    # 実験を行いたい部位（ROI）のリスト
    # マスクファイル名が w{subject}_mask-{ROI}.nii となることを想定
    roi_list = [
        'temporal-lobe',
    ]

    # 実験を行いたい出力指標（Target）のリスト
    target_list = [
        'MMSE',
    ]

    # ベースとなるconfigファイルのパス
    config_path = 'src/config/config.yaml'
    
    # メインスクリプトのパス
    main_script_path = 'src/main.py'

    # ------------------

    total_experiments = len(roi_list) * len(target_list)
    current_count = 0

    print(f"=== 合計 {total_experiments} 件の実験を開始します ===")

    for roi in roi_list:
        for target in target_list:
            current_count += 1
            print(f"\n\n{'='*60}")
            print(f"Experiment {current_count}/{total_experiments}")
            print(f"ROI: {roi} | Target: {target}")
            print(f"{'='*60}\n")

            # コマンドの構築
            # python src/main.py --config ... --roi ... --target ...
            cmd = [
                sys.executable,  # 現在のPythonインタプリタのパス
                main_script_path,
                '--config', config_path,
                '--roi', roi,
                '--target', target
            ]

            try:
                # サブプロセスとして実行
                result = subprocess.run(cmd, check=True)
                
                print(f"\n>> Experiment {current_count} Finished Successfully.")
                
            except subprocess.CalledProcessError as e:
                print(f"\n!! Experiment {current_count} Failed with error code {e.returncode}.")
                # エラーが出ても次の実験に進む場合は continue
                # 全停止したい場合は raise e
                continue
            
            # GPUを休ませる場合などに少し待機を入れることも可能
            time.sleep(2)

    print("\n=== すべての実験ループが終了しました ===")

if __name__ == '__main__':
    run_experiments()