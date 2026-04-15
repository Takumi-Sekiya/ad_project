import copy
import numpy as np

class EarlyStopping:
    """
    KerasのEarlyStoppingをPyTorchで再現するクラス
    """
    def __init__(self, patience=7, verbose=1, restore_best_weights=True):
        self.patience = patience
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            # ベストスコアが更新されたら重みを保存
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience} (Best: {self.best_loss:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True

    def restore(self, model):
        """保存しておいたベストな重みをモデルに復元する"""
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print("Restored best model weights from EarlyStopping.")