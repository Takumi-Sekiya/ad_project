import tensorflow as tf
from typing import List, Dict, Any

def create_callbacks_from_config(config_list: List[Dict[str, Any]]) -> List[tf.keras.callbacks.Callback]:
    """
    設定ファイルのリストから, Kerasのコールバックオブジェクトのリストを生成する
    """
    if not config_list:
        return []
    
    callbacks = []
    for cb_config in config_list:
        cb_name = cb_config['name']
        cb_params = cb_config.get('params', {})

        # 文字列の名前から, tf.keras.callbacksモジュールのクラスを取得
        try:
            CallbackClass = getattr(tf.keras.callbacks, cb_name)
        except AttributeError:
            raise ValueError(f"Callback '{cb_name}' not found in tf.keras.callbacks.")
        
        # パラメータ付きでクラスをインスタンス化
        callbacks.append(CallbackClass(**cb_params))

    return callbacks