import pickle 
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

class AlzheimerDataset(Dataset):
    """
    PyTorch用のカスタムデータセットクラス
    """
    def __init__(self, img_data, tabular_data=None, labels=None):
        # TensorFlow: (Batch, X, Y, Z, 1) -> PyTorch: (Batch, 1, X, Y, Z) に次元を入れ替える
        if img_data.ndim == 5 and img_data.shape[-1] == 1:
            img_data = np.transpose(img_data, (0, 4, 1, 2, 3))
        elif img_data.ndim == 4:
            # もし (Batch, X, Y, Z) だった場合はChannel次元を追加
            img_data = np.expand_dims(img_data, axis=1)
            
        self.img_data = torch.tensor(img_data, dtype=torch.float32)
        
        self.tabular_data = None
        if tabular_data is not None:
            self.tabular_data = torch.tensor(tabular_data, dtype=torch.float32)
            
        self.labels = None
        if labels is not None:
            # 回帰タスクのため、ラベルも(Batch, 1)の形にする
            self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        sample = {'img_input': self.img_data[idx]}
        if self.tabular_data is not None:
            sample['numerical_input'] = self.tabular_data[idx]
            
        if self.labels is not None:
            return sample, self.labels[idx]
        return sample

def get_datasets(config: dict):
    """
    Pickleファイルからデータを読み込み, 訓練用とテスト用の Dataset を返す
    ※DataLoader化は engine.py 側で行います。
    """
    pickle_path = Path.home() / "ad_project" / config['data']['pickle_path']
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    target_name = config['data']['target_variable']
    tabular_features_list = config['data'].get('tabular_input_features', [])

    y_train = data['features_train'][target_name].values
    y_test = data['features_test'][target_name].values

    X_train_img = data['img_train']
    X_test_img = data['img_test']

    if tabular_features_list:
        print(f"Multimodal mode enabled. Using features: {tabular_features_list}")
        X_train_tabular = data['features_train'][tabular_features_list].values.astype(np.float32)
        X_test_tabular = data['features_test'][tabular_features_list].values.astype(np.float32)

        train_ds = AlzheimerDataset(X_train_img, tabular_data=X_train_tabular, labels=y_train)
        test_ds = AlzheimerDataset(X_test_img, tabular_data=X_test_tabular, labels=y_test)
    else:
        print("Unimodal mode enabled (image only).")
        train_ds = AlzheimerDataset(X_train_img, labels=y_train)
        test_ds = AlzheimerDataset(X_test_img, labels=y_test)

    return train_ds, test_ds