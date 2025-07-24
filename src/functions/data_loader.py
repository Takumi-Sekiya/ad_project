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

    # Pickleファイルからデータの取得
    y_train = data['features_train'][target_name]
    y_test = data['features_test'][target_name]

    X_train_img = data['img_train']
    X_test_img = data['img_test']

    train_ds = tf.data.Dataset.from_tensor_slices((X_train_img, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test_img, y_test))

    return train_ds, test_ds