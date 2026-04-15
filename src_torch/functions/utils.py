import os
import random
import numpy as np
import torch

def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # PyTorch用のシード固定
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # cuDNNの決定論的動作を保証（再現性のため）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False