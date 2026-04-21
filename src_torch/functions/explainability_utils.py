# explainability_utils.py
import json
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

def load_and_preprocess_mask(mask_path, threshold=50):
    """data_handling.py と完全に同一のマスク前処理関数"""
    mask_nii = nib.load(mask_path)
    mask_data = mask_nii.get_fdata().astype(np.float32)
    
    # 0-255などのスケールの場合、閾値以上を1.0にする
    if mask_data.max() > 1.0:
        mask_data = np.where(mask_data >= threshold, 1.0, 0.0)
        
    return mask_data

def crop_by_ranges(array, ranges):
    """JSONから取得した絶対座標で画像を切り出す"""
    return array[ranges[0][0]:ranges[0][1],
                 ranges[1][0]:ranges[1][1],
                 ranges[2][0]:ranges[2][1]]

def normalize_intensity(array):
    min_val, max_val = array.min(), array.max()
    if max_val - min_val > 0:
        return (array - min_val) / (max_val - min_val)
    return array

def load_crop_metadata(json_path, roi_name):
    """JSONファイルから対象ROIの切り出し座標とキャンバスサイズを取得する"""
    print(f"Loading crop metadata from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if roi_name not in data:
        raise ValueError(f"ROI '{roi_name}' のデータがJSONに見つかりません。")
        
    ranges_dict = data[roi_name]['crop_ranges']
    ranges = [ranges_dict['x'], ranges_dict['y'], ranges_dict['z']]
    target_shape = tuple(data[roi_name]['target_shape'])
    
    return ranges, target_shape

def generate_model_input_reproduction(base_path, mask_path, crop_ranges):
    """固定座標と、学習時と同一のマスク前処理を用いて入力テンソルを再現する"""
    base_nii = nib.load(base_path)
    base_data = base_nii.get_fdata().astype(np.float32)
    
    mask_data = load_and_preprocess_mask(mask_path, threshold=50)
    
    roi_array = base_data * (mask_data > 0.5)
    
    processed_img = crop_by_ranges(roi_array, crop_ranges)
    processed_img = normalize_intensity(processed_img)
        
    return processed_img

# ==========================================
# 2. データロード & 処理用ユーティリティ
# ==========================================
"""
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
"""
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
    """文字列で指定されたターゲットレイヤーをPyTorchモデルから取得する"""
    if layer_name is None:
        if hasattr(model, 'layer8'):
            # ResNet3D の場合: 最後のブロック全体
            return [model.layer8]
        elif hasattr(model, 'layer6'):
            # RegulatedResNet3D の場合: 最後のブロック全体
            return [model.layer6]
        elif hasattr(model, 'conv5'):
            # Simple3DCNN / Multimodal3DCNN の場合: 
            # conv5は Sequential(Conv3d, ReLU, Pool) なので、空間解像度が潰れる前の Conv3d [0] を狙う
            return [model.conv5[0]]
        else:
            raise ValueError("Could not automatically determine target layer.")
    else:
        # 'layer6' のように文字列で直接指定された場合
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