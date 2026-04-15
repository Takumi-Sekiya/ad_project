# explainability_utils.py
import pickle
import numpy as np
import nibabel as nib
from pathlib import Path
import torch

# pytorch-grad-cam のインポート
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==========================================
# 1. image_preprocessing.py のロジック (完全再現)
# ==========================================

def crop_roi(roi_array):
    if np.all(roi_array == 0):
        return np.zeros((1, 1, 1), dtype=roi_array.dtype)
    indeces = np.array(np.where(roi_array != 0))
    min_coords = indeces.min(axis=1)
    max_coords = indeces.max(axis=1)
    return roi_array[min_coords[0]:max_coords[0]+1,
                     min_coords[1]:max_coords[1]+1,
                     min_coords[2]:max_coords[2]+1]

def pad_to_canvas(cropped_array, canvas_shape):
    canvas = np.zeros(canvas_shape, dtype=cropped_array.dtype)
    start_coords = [(cs - da) // 2 for cs, da in zip(canvas_shape, cropped_array.shape)]
    end_coords = [sc + da for sc, da in zip(start_coords, cropped_array.shape)]
    canvas[start_coords[0]:end_coords[0],
              start_coords[1]:end_coords[1],
              start_coords[2]:end_coords[2]] = cropped_array
    return canvas

def normalize_intensity(array):
    min_val, max_val = array.min(), array.max()
    if max_val - min_val > 0:
        return (array - min_val) / (max_val - min_val)
    return array

# ==========================================
# 2. データロード & 処理用ユーティリティ
# ==========================================

def get_canvas_shape_from_pickle(pickle_path):
    print(f"Loading pickle to determine canvas size: {pickle_path}")
    pickle_path = Path(pickle_path)
    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    if 'img_train' not in data:
        raise ValueError("Pickle file does not contain 'img_train' key.")
    
    img_shape = data['img_train'].shape
    
    # Keras (N, X, Y, Z, 1) or (N, X, Y, Z) の形式からキャンバスサイズを取得
    # ※保存されているPickle自体の形式は変わっていない前提
    if len(img_shape) == 5:
        canvas_shape = img_shape[1:4]
    elif len(img_shape) == 4:
        canvas_shape = img_shape[1:4]
    else:
        raise ValueError(f"Unexpected image shape in pickle: {img_shape}")
    
    print(f"Detected canvas shape from pickle: {canvas_shape}")
    return canvas_shape

def generate_model_input_reproduction(base_path, mask_path, canvas_shape):
    base_nii = nib.load(base_path)
    mask_nii = nib.load(mask_path)
    
    base_data = base_nii.get_fdata().astype(np.float32)
    mask_data = mask_nii.get_fdata().astype(np.float32)
    
    roi_array = base_data * (mask_data > 0.5)
    cropped = crop_roi(roi_array)
    padded = pad_to_canvas(cropped, canvas_shape)
    processed_img = normalize_intensity(padded)
        
    return processed_img

# ==========================================
# 3. Grad-CAMロジック (PyTorch移行)
# ==========================================

class RegressionScoreTarget:
    """
    回帰タスク用のターゲット関数
    mode='increase' の場合は予測値をそのまま最大化
    mode='decrease' の場合は予測値を最小化（符号を反転）
    """
    def __init__(self, mode='increase'):
        self.mode = mode
        
    def __call__(self, model_output):
        if self.mode == 'increase':
            return model_output[0]
        else:
            return -1.0 * model_output[0]

def load_exported_model_weights(model_instance, model_path):
    """
    保存されたPyTorchの重み (.pth) を読み込む
    ※事前に model_instance (build_modelで生成したもの) を渡す必要がある
    """
    path_obj = Path(model_path)
    
    if path_obj.is_dir():
         raise ValueError(f"Error: Expected a .pth file, but got a directory: {model_path}")
    
    print(f"Loading PyTorch weights from: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.to(device)
    model_instance.eval()
    
    return model_instance

def get_target_layer(model, layer_name):
    """文字列で指定されたターゲットレイヤーをPyTorchモデルから取得する簡易関数"""
    # 例: 'layer4' -> model.layer4 など。設定に応じて調整が必要です。
    # Noneが渡された場合は、最後の畳み込み層を推測して返すようにします。
    if layer_name is None:
        # Simple3DCNN や Multimodal の場合
        if hasattr(model, 'conv5'):
            return [model.conv5[-2]] # -1はPoolingなのでその前のConvを狙う
        # ResNet の場合
        elif hasattr(model, 'layer8'):
            return [model.layer8.conv2]
        elif hasattr(model, 'layer6'):
            return [model.layer6.conv2]
        else:
            raise ValueError("Could not automatically determine target layer. Please specify explicitly.")
    else:
        # getattrなどで取得するロジック（必要に応じて実装）
        return [getattr(model, layer_name)]

def compute_3d_gradcam(model, img_array, numerical_array=None, score_mode='increase', target_layer_name=None):
    """3D Grad-CAMの計算 (pytorch-grad-camを使用)"""
    device = next(model.parameters()).device
    
    # 入力配列の整形 (1, 1, X, Y, Z)
    input_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # マルチモーダルの場合、pytorch-grad-camは複数入力にデフォルトで対応していないため工夫が必要
    # ※今回はシンプル化のため、画像のみのモデルを前提としています。
    if numerical_array is not None:
         print("Warning: Grad-CAM with multimodal inputs requires custom wrapper. Running image-only flow.")

    target_layers = get_target_layer(model, target_layer_name)
    
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [RegressionScoreTarget(mode=score_mode)]

    # grayscale_cam は (1, X, Y, Z) の形式で返る
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    heatmap = grayscale_cam[0, :]
    
    return heatmap

def save_nifti(data, output_path):
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, output_path)
    print(f"Saved: {output_path}")