# utils.py
"""
補助関数モジュール.
シェルコマンドの実行など, 共通の処理を定義する.
"""

import subprocess
import shlex

def run_command(command_string, log_file=None):
    """
    シェルコマンドを実行し, 標準出力と標準エラーを処理する.

    Args:
        command_string (str): 実行するシェルコマンドの文字列.
        log_file (Path or str, optional): 出力ログを書き込むファイルパス.

    Returns:
        bool: コマンドが成功した場合はTrue, 失敗した場合はFalse.
    """
    print(f"Executing command: {command_string}")

    try:
        if log_file:
            full_command = f"{command_string} > {log_file} 2>&1"
        else:
            full_command = command_string

        result = subprocess.run(
            full_command,
            shell=True,
            executable='/bin/bash',
            check=True
        )

        print("Command executed successfully.: ", command_string)
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command_string}")
        print(f"Return code: {e.returncode}")
        if log_file:
            print(f"Check log file for details: {log_file}")
        return False
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False