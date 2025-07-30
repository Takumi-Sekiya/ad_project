import pickle 
import numpy as np
from pathlib import Path
import tensorflow as tf

def get_datasets(config: dict):
    """
    Pickleファイルからデータを読み込み, 訓練用とテスト用のtf.data.Datasetを返す
    """

    # Pickleファイルをロード
    pickle_path = Path.home() / "ad_project" / config['data']['pickle_path']
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    # 目的変数名の取得
    target_name = config['data']['target_variable']
    tabular_features_list = config['data'].get('tabular_input_features', [])

    # Pickleファイルからデータの取得
    y_train = data['features_train'][target_name].values
    y_test = data['features_test'][target_name].values

    X_train_img = data['img_train']
    X_test_img = data['img_test']

    # マルチモーダルな場合の処理
    if tabular_features_list:
        print(f"Multimodal mode enabled. Using features: {tabular_features_list}")

        X_train_tabular = data['features_train'][tabular_features_list].values.astype(np.float32)
        X_test_tabular = data['features_test'][tabular_features_list].values.astype(np.float32)

        train_ds = tf.data.Dataset.from_tensor_slices((
            {'img_input': X_train_img, 'numerical_input': X_train_tabular},
            y_train
        ))
        test_ds = tf.data.Dataset.from_tensor_slices((
            {'img_input': X_test_img, 'numerical_input': X_test_tabular},
            y_test
        ))

    else:
        print("unimodal mode enabled (image only).")
        train_ds = tf.data.Dataset.from_tensor_slices((X_train_img, y_train))
        test_ds = tf.data.Dataset.from_tensor_slices((X_test_img, y_test))

    return train_ds, test_ds