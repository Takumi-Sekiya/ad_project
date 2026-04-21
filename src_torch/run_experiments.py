# python src_torch/run_experiments.py

import subprocess
import sys
import time

def run_experiments():
    # --- 設定エリア ---
    
    # 実験を行いたい部位（ROI）のリスト
    # マスクファイル名が w{subject}_mask-{ROI}.nii となることを想定
    roi_list = [
        #'gray-matter',
        'frontal-lobe',
        'parietal-lobe',
        'occipital-lobe',
        'temporal-lobe',
        'subcortical-tissue',
        'superior-frontal',           # 上前頭回
        'rostral-middle-frontal',     # 吻側中前頭回
        'caudal-middle-frontal',      # 尾側中前頭回
        'pars-opercularis',           # 弁蓋部
        'pars-triangularis',          # 三角部
        'pars-orbitalis',             # 眼窩部
        'lateral-orbitofrontal',      # 外側眼窩前頭回
        'medial-orbitofrontal',       # 内側眼窩前頭回
        'frontal-pole',               # 前頭極
        'rostral-anterior-cingulate', # 吻側前帯状回
        'caudal-anterior-cingulate',  # 尾側前帯状回
        'precentral',                 # 中心前回
        # --- Prietal Lobe ---
        'inferior-parietal',          # 下頭頂小葉
        'postcentral',                # 中心後回
        'precuneus',                  # 楔前部
        'superior-parietal',          # 上頭頂小葉
        'supramarginal',              # 縁上回
        # --- Occipital Lobe ---
        'cuneus',                     # 楔部
        'lateral-occipital',          # 外側後頭皮質
        'lingual',                    # 舌状回
        'pericalcarine',              # 鳥距溝周囲
        # --- Temporal Lobe ---
        'superior-temporal',          # 上側頭回
        'middle-temporal',            # 中側頭回
        'inferior-temporal',          # 下側頭回
        'banks-sts',                  # 上側頭溝堤
        'fusiform',                   # 紡錘状回
        'transverse-temporal',        # 横側頭回
        'entorhinal',                 # 嗅内野
        'parahippocampal',            # 海馬傍回
        'temporal-pole',              # 側頭極
        # --- Other Cortical Regions ---
        'posterior-cingulate',        # 後部帯状回 (帯状回)
        'isthmuscingulate',           # 帯状回峡部
        'paracentral',                # 傍中心小葉
        'insula',                     # 島皮質
        # --- Subcortical Regions ---
        'whole-hippocampus',          # 海馬
        'amygdala',                   # 扁桃体
        'brain-stem',                 # 脳幹
        'thalamus',                   # 視床
        'caudate',                    # 尾状核
        'putamen',                    # 被殻
        'pallidum',                   # 淡蒼球
        'accumbens-area',             # 側坐核
        'ventralDC',                  # 腹側間脳
        'cerebellum',                 # 小脳
    ]

    # 実験を行いたい出力指標（Target）のリスト
    target_list = [
        #'MMSE',
        #'HDSR',
        #'gm_atrophy',
        'scaled_MMSE',
        'scaled_HDSR',
        'scaled_gm_atrophy',
        #'scaled_severity'
    ]

    # ベースとなるconfigファイルのパス
    config_path = 'src_torch/config/config.yaml'
    
    # メインスクリプトのパス
    main_script_path = 'src_torch/main_cross.py'

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
            # python src_torch/main.py --config ... --roi ... --target ...
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

