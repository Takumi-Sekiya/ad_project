# explainability_utils.py
import pickle
import numpy as np
import tensorflow as tf
import nibabel as nib
from pathlib import Path
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import Score

# ==========================================
# 1. image_preprocessing.py のロジック (完全再現)
# ==========================================

def crop_roi(roi_array):
    """
    ROIの周囲の0埋めの部分を削除する (image_preprocessing.pyより)
    """
    if np.all(roi_array == 0):
        # ROIがない場合は1x1x1のゼロ配列を返す
        return np.zeros((1, 1, 1), dtype=roi_array.dtype)
    indeces = np.array(np.where(roi_array != 0))
    min_coords = indeces.min(axis=1)
    max_coords = indeces.max(axis=1)
    return roi_array[min_coords[0]:max_coords[0]+1,
                     min_coords[1]:max_coords[1]+1,
                     min_coords[2]:max_coords[2]+1]

def pad_to_canvas(cropped_array, canvas_shape):
    """
    指定されたcanvas_shapeの中央に画像を配置する (image_preprocessing.pyより)
    """
    canvas = np.zeros(canvas_shape, dtype=cropped_array.dtype)
    start_coords = [(cs - da) // 2 for cs, da in zip(canvas_shape, cropped_array.shape)]
    end_coords = [sc + da for sc, da in zip(start_coords, cropped_array.shape)]
    canvas[start_coords[0]:end_coords[0],
              start_coords[1]:end_coords[1],
              start_coords[2]:end_coords[2]] = cropped_array
    return canvas

def normalize_intensity(array):
    """
    画像の輝度値から[0, 1]の範囲に正規化する (image_preprocessing.pyより)
    """
    min_val, max_val = array.min(), array.max()
    if max_val - min_val > 0:
        return (array - min_val) / (max_val - min_val)
    return array

# ==========================================
# 2. データロード & 処理用ユーティリティ
# ==========================================

def get_canvas_shape_from_pickle(pickle_path):
    """Pickleファイルから学習データの形状を取得する"""
    print(f"Loading pickle to determine canvas size: {pickle_path}")
    pickle_path = Path(pickle_path)
    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    if 'img_train' not in data:
        raise ValueError("Pickle file does not contain 'img_train' key.")
    
    img_shape = data['img_train'].shape
    
    # (N, X, Y, Z, 1) -> (X, Y, Z)
    if len(img_shape) == 5:
        canvas_shape = img_shape[1:4]
    elif len(img_shape) == 4:
        canvas_shape = img_shape[1:4]
    else:
        raise ValueError(f"Unexpected image shape in pickle: {img_shape}")
    
    print(f"Detected canvas shape from pickle: {canvas_shape}")
    return canvas_shape

def generate_model_input_reproduction(base_path, mask_path, canvas_shape):
    """モデル入力用の画像を生成する (data_handling.py再現)"""
    base_nii = nib.load(base_path)
    mask_nii = nib.load(mask_path)
    
    base_data = base_nii.get_fdata().astype(np.float32)
    mask_data = mask_nii.get_fdata().astype(np.float32)
    
    # 1. マスク適用
    roi_array = base_data * (mask_data > 0.5)
    # 2. Crop
    cropped = crop_roi(roi_array) # 上部で定義されている前提
    # 3. Pad
    padded = pad_to_canvas(cropped, canvas_shape) # 上部で定義されている前提
    # 4. Normalize
    processed_img = normalize_intensity(padded) # 上部で定義されている前提
        
    return processed_img

# ==========================================
# 3. Grad-CAMロジック (モデル読み込み修正)
# ==========================================

class RegressionScore(Score):
    def __init__(self, mode='increase'):
        self.mode = mode
    def __call__(self, output):
        if self.mode == 'increase':
            return output[:, 0]
        else:
            return -1.0 * output[:, 0]

def load_exported_model(model_path):
    """
    Keras 3対応のモデル読み込み関数
    .keras ファイルであることを確認して読み込む
    """
    path_obj = Path(model_path)
    
    # ディレクトリが指定された場合のエラーハンドリング
    if path_obj.is_dir():
        raise ValueError(
            f"Error: The provided path '{model_path}' is a directory.\n"
            "Keras 3 (TF 2.16+) no longer supports loading SavedModel directories via load_model.\n"
            "Please modify your training script (engine.py) to save the model as a '.keras' file "
            "using `model.save('model.keras', save_format='keras')`."
        )
    
    # 拡張子のチェック (警告のみ)
    if path_obj.suffix not in ['.keras', '.h5']:
        print(f"Warning: Model file extension is '{path_obj.suffix}'. Keras 3 recommends '.keras'.")

    print(f"Loading model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from {model_path}.\n"
            f"Ensure it is a valid Keras model file (.keras or .h5).\nOriginal Error: {e}"
        )

def compute_3d_gradcam(model, img_array, score_mode='increase', target_layer=None):
    """3D Grad-CAMの計算"""
    replace2linear = ReplaceToLinear()
    
    # モデル構造のチェック
    if not isinstance(model, tf.keras.Model):
         raise ValueError("The loaded object is not a valid Keras Model. Ensure it was saved/loaded correctly.")

    gradcam = Gradcam(model, model_modifier=replace2linear, clone=True)
    score = RegressionScore(mode=score_mode)
    
    input_tensor = np.expand_dims(img_array, axis=0) 
    if input_tensor.ndim == 4:
        input_tensor = np.expand_dims(input_tensor, axis=-1)

    cam = gradcam(score, input_tensor, penultimate_layer=target_layer, seek_penultimate_conv_layer=True)
    heatmap = cam[0]
    
    heatmap = normalize_intensity(heatmap)
    return heatmap

def save_nifti(data, output_path):
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, output_path)
    print(f"Saved: {output_path}")