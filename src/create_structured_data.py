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

def main():
    """
    複数の生エクセルデータからマッピングファイルに基づき, 単一の構造化csvファイルを生成する.
    """
    print("処理を開始 ...")

    # 1. マッピングファイルの読み込み
    try:
        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        source_data_list = config.get("source_data", {})
        final_columns_info = config.get("final_columns", {})
        if not source_data_list or not final_columns_info:
            print(f"エラー: {MAPPING_FILE} の形式が正しくありません. 'source_data' と 'final_columns' のキーが必要です.")
            sys.exit(1)

    except FileNotFoundError:
        print(f"エラー: マッピングファイル {MAPPING_FILE} が見つかりません。")
        sys.exit(1)

    # 整形後のDataFrameを格納するリスト
    processed_dfs = []

    # 2. ファイルごとに整形処理
    print("ファイルの読み込み, 整形 ...")
    for source_info in source_data_list:
        filename = source_info.get("filename")
        sheet_name = source_info.get("sheet_name", 0)
        column_map = source_info.get("columns")

        if not filename or not column_map:
            print(f"警告: 'filename' または 'columns' の設定が不完全な項目があります。スキップします。")
            continue

        file_path = RAW_DATA_DIR / filename
        if not file_path.exists():
            print(f"警告: {file_path} が見つかりません. スキップします.")
            continue

        print(f"処理中: {filename} (シート: {sheet_name if sheet_name != 0 else '最初のシート'})")
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            print(f"  エラー: {filename} のシート '{sheet_name}' の読み込みに失敗しました。詳細: {e}")
            continue

        # マッピングを反転
        rename_dict = {v: k for k, v in column_map.items()}

        # 属性名の変更. 存在しない属性は無視.
        original_cols_to_use = list(rename_dict.keys())
        df_renamed = df[original_cols_to_use].rename(columns=rename_dict)

        # 存在しない列をNaNで追加
        final_cols_list = list(final_columns_info.keys())
        df_final = df_renamed.reindex(columns=final_cols_list)

        processed_dfs.append(df_final)

    if not processed_dfs:
        print("エラー: 処理できるデータがありませんでした. 処理を中断します.")
        sys.exit(1)

    # 3. 全データの統合
    print("データの統合 ...")
    combined_df = pd.concat(processed_dfs, ignore_index=True)

    # 4. データ型の統一とクレンジング
    print("データ型の統一 ...")
    for col_name, col_info in final_columns_info.items():
        if col_info.get("type") == "numeric" and col_name in combined_df.columns:
            combined_df[col_name] = pd.to_numeric(combined_df[col_name], errors='coerce')
    
    # 5. IDの重複のチェック
    print("IDの重複のチェック ...")
    if 'subject_id' in combined_df.columns:
        id_series = combined_df['subject_id'].dropna()
        duplicated_ids = id_series[id_series.duplicated()].unique()

        if len(duplicated_ids) > 0:
            print("subject_idに重複を発見.")
            print(list(duplicated_ids))
            print("処理を中断. データとマッピングファイルを確認してください.")
            sys.exit(1)
        else:
            print("  IDの重複なし.")
        
    else:
        print("警告: 'parent_id'カラムが見つからないため, 重複チェックをスキップ.")
    
    # 6. 出力
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print("\n--- 処理完了 ---")
    print(f"構造化データを{OUTPUT_FILE}に保存.")
    print("\n最終データフレームの情報:")
    combined_df.info()

if __name__ == '__main__':
    main()