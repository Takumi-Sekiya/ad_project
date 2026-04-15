import pandas as pd
import yaml
import json
from pathlib import Path
import sys

# --- 設定 ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw_data"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MAPPING_FILE = BASE_DIR / "src_torch" / "config" / "mapping.yaml"
OUTPUT_FILE = PROCESSED_DATA_DIR / "structured_data.csv"

def load_config(mapping_file_path):
    """
    設定ファイルを読み込み, 内容を検証する
    """
    try:
        with open(mapping_file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        sources = config.get("sources", [])
        final_columns_info = config.get("final_columns", [])
        if not sources or not final_columns_info:
            print(f"エラー: {mapping_file_path}の'sources'または'final_columns'が不適切です.")  
            sys.exit(1)
        return sources, final_columns_info

    except FileNotFoundError:
        print(f"エラー: マッピングファイル{mapping_file_path}が見つかりません.")
        sys.exit(1)         

def process_source_file(source_info, raw_data_dir):
    """
    単一のデータソースファイルを読み込み, 設定に基づいて整形する.
    """
    filename = source_info.get("file")
    file_path = raw_data_dir / filename

    if not file_path.exists():
        print(f" - 警告: {file_path}が見つかりません. スキップします.")
        return None
    
    # ファイル読み込み
    try:
        sheet = source_info.get("sheet", 0)
        skip_after_header = source_info.get("skip_after_header", 0)
        rows_to_skip = list(range(1, skip_after_header + 1)) if skip_after_header > 0 else None

        if file_path.suffix == '.xlsx':
            df = pd.read_excel(file_path, sheet_name=sheet, header=0, skiprows=rows_to_skip, engine='openpyxl')
        elif file_path.suffix == '.xls':
            df = pd.read_excel(file_path, sheet_name=sheet, header=0, skiprows=rows_to_skip, engine='xlrd')
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path, header=0, skiprows=rows_to_skip)
        else:
            print(f" - 警告: 未対応のファイル形式です: {file_path.name}")
            return None
    except Exception as e:
        print(f"エラー: ファイル読み込みに失敗しました。\n詳細: {e}")
        return None
        
    # カラム整形
    rename_dict = {v: k for k, v in source_info.get("columns", {}).items()}
    if 'subject_id' not in rename_dict.values():
        print(f" - 警告: {filename}のマッピングに 'subject_id' がありません. スキップします.")
        return None
        
    df.rename(columns=rename_dict, inplace=True)

    cols_to_skip = [col for col in rename_dict.values() if col in df.columns]
    if not cols_to_skip:
        return None
    df = df[cols_to_skip]

    # データクレンジング
    df.dropna(subset=['subject_id'], inplace=True)
    if df.empty:
        return None

    df['subject_id'] = df['subject_id'].astype(str)
    df.set_index('subject_id', inplace=True)
        
    return df

def finalize_and_save_data(df, final_columns_info, output_path):
    """
    最終的なDataFrameを整形、正規化し、CSVとして保存する.
    """
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'subject_id'}, inplace=True)

    scaling_metadata = {}

    # 1. データ型の統一とスケーリング処理
    for col_name, col_info in final_columns_info.items():
        # 数値型へのキャスト
        if col_name in df.columns and col_info.get("type") == 'numeric':
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

            # スケーリングの処理
            if col_info.get("scale") is True:
                min_val = df[col_name].min()
                max_val = df[col_name].max()
                invert = col_info.get("invert", False)
                scaled_col_name = f"scaled_{col_name}"

                # 欠損値しかない、または最大値と最小値が等しい（ゼロ除算防止）場合のハンドリング
                if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
                    print(f" - 警告: カラム '{col_name}' は有効な分散がないためスケーリングをスキップ、または0埋めします.")
                    df[scaled_col_name] = 0.0
                else:
                    # 正規化 (0〜1)
                    df[scaled_col_name] = (df[col_name] - min_val) / (max_val - min_val)
                    # 反転処理
                    if invert:
                        df[scaled_col_name] = 1.0 - df[scaled_col_name]

                # メタデータの記録 (NaNの場合は記録のためにNoneに変換)
                scaling_metadata[col_name] = {
                    "original_column": col_name,
                    "scaled_column": scaled_col_name,
                    "min": float(min_val) if not pd.isna(min_val) else None,
                    "max": float(max_val) if not pd.isna(max_val) else None,
                    "inverted": invert
                }

    # 2. カラムの順序を整理 (元のfinal_columns + 追加されたscaledカラム + その他)
    final_order = [col for col in final_columns_info.keys() if col in df.columns]
    scaled_cols = [f"scaled_{col}" for col in scaling_metadata.keys()]
    other_cols = [col for col in df.columns if col not in final_order and col not in scaled_cols]
    df = df[final_order + scaled_cols + other_cols]

    # 3. データの出力
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"統合後データが {output_path} に保存されました.")

    # 4. スケーリング用メタデータの保存（設定がある場合のみ）
    if scaling_metadata:
        metadata_path = output_path.parent / "scaling_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(scaling_metadata, f, indent=4, ensure_ascii=False)
        print(f"スケーリング情報（メタデータ）が {metadata_path} に保存されました.")

    print(f"合計 {len(df)} 件のユニークIDが処理されました.")
    df.info()

def main():
    """
    複数のデータソースをIDベースでマージし, 単一のマスターデータを生成する.
    """
    sources, final_columns_info = load_config(MAPPING_FILE)
    master_df = pd.DataFrame()

    for i, source in enumerate(sources):
        filename = source.get("file", "N/A")
        print(f" - 処理中 ({i+1}/{len(sources)}): {filename}")

        temp_df = process_source_file(source, RAW_DATA_DIR)
        
        if temp_df is None:
            print(f" - 警告: {filename} から有効な情報は得られませんでした.")
            continue
        
        if master_df.empty:
            master_df = temp_df
        else:
            # 【重要】updateからcombine_firstへの修正（欠損値を互いに補完し、新規カラムもマージする）
            master_df = temp_df.combine_first(master_df)

    finalize_and_save_data(master_df, final_columns_info, OUTPUT_FILE)

if __name__ == '__main__':
    main()