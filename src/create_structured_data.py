import pandas as pd
import yaml
from pathlib import Path
import sys

# --- 設定 ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw_data"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MAPPING_FILE = BASE_DIR / "src" / "config" / "mapping.yaml"
OUTPUT_FILE = PROCESSED_DATA_DIR / "structured_data.csv"

def load_config(mapping_file_path):
    """
    設定ファイルを読み込み, 内容を検証する
    """
    try:
        with open(mapping_file_path, 'r', encodinf='utf-8') as f:
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

        if file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, sheet_name=sheet, header=0, skiprows=rows_to_skip, engine='openpyxl')
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path, header=0, skiprows=rows_to_skip)
        else:
            print(f" - 警告: 未対応のファイル形式です: {file_path.name}")
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
        if df.empty():
            return None

        df['subject_id'] = df['subject_id'].astype(str)
        df.set_index('subject_id', inplace=True)
        
        return df

def finalize_and_save_data(df, final_columns_info, output_path):
    """
    最終的なDataFrameを整形し, CSVとして保存する.
    """
    if df.empty():
        print(f"エラー: 処理結果が空です. 処理を中止します.")
        sys.exit(1)

    df.reset_index(inplace=True)

    # データ型の統一
    for col_name, col_info in final_columns_info.items():
        if col_name in df.columns and col_info.get("type") == 'numeric':
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

    # カラムの順序を final_columns に揃える
    final_order = [col for col in final_columns_info.keys() if col in df.columns]
    other_cols = [col for col in df.columns if col not in final_order]
    df = df[final_order + other_cols]

    # 出力
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"統合後データが {output_path} に保存されました.")
    print(f"合計 {len(df)} 件のユニークIDが処理されました.")

def main():
    """
    複数のデータソースをIDベースでマージし, 単一のマスターデータを生成する.
    """
    # 1. 設定の読み込み
    source, final_columns_info = load_config(MAPPING_FILE)

    # 2. マスターDataFrameの初期化
    master_df = pd.DataFrame()

    # 3. 各データソースを処理し, マスターへマージ
    for i, source in enumerate(sources):
        filename = source.get("file", "N/A")
        print(f" - 処理中 ({i+1}/{len(sources)}): {filename}")

        temp_df = process_source_file(source, RAW_DATA_DIR)
        
        if temp_df is None:
            print(f" - 警告: {filename} から有効な情報は得られませんでした.")
            continue
        
        if master_df.empty():
            master_df = temp_df
        else:
            master_df.update(temp_df)
            new_ids = temp_df.index.difference(master_df.index)
            if not new_ids.empty():
                master_df = pd.concat([master_df, temp_df.loc[new_ids]])

    finalize_and_save_data(master_df, final_columns_info, OUTPUT_FILE)

if __name__ == '__main__':
    main()