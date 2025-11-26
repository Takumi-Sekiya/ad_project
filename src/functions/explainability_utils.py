# explainability_utils.py
import numpy as np
import tensorflow as tf
import nibabel as nib
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import Score
from nilearn import datasets, image
from nilearn.input_data import NiftiLabelsMasker
import math

# ==========================================
# 前処理・逆変換ロジック (data_handling.py 準拠)
# ==========================================

def get_bbox(mask_array):
    """マスクからバウンディングボックス(最小包含矩形)の座標を取得"""
    rows = np.any(mask_array, axis=(1, 2))
    cols = np.any(mask_array, axis=(0, 2))
    slices = np.any(mask_array, axis=(0, 1))
    
    if not np.any(rows):
        raise ValueError("Mask is empty.")

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    smin, smax = np.where(slices)[0][[0, -1]]

    return (rmin, rmax+1), (cmin, cmax+1), (smin, smax+1)

def pad_to_canvas(img, canvas_shape):
    """画像をキャンバスサイズに合わせてパディング (data_handling.pyより再現)"""
    # 実際の実装に合わせて調整が必要ですが、ここでは中央寄せパディングを想定
    dataset_shape = img.shape
    pads = []
    for d_img, d_canvas in zip(dataset_shape, canvas_shape):
        pad_total = max(0, d_canvas - d_img)
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        pads.append((pad_before, pad_after))
    
    # パディング実行
    padded_img = np.pad(img, pads, mode='constant', constant_values=0)
    
    # キャンバスサイズより大きい場合はクロップ (念のため)
    slices = tuple(slice(0, d) for d in canvas_shape)
    return padded_img[slices], pads

def generate_input_data(base_path, mask_path, canvas_shape, mode='crop_and_pad'):
    """
    BaseとMaskからモデル入力画像を生成し、逆変換に必要な座標情報を返す
    """
    base_nii = nib.load(base_path)
    mask_nii = nib.load(mask_path)
    
    base_data = base_nii.get_fdata().astype(np.float32)
    mask_data = mask_nii.get_fdata().astype(np.float32)
    affine = base_nii.affine
    original_shape = base_data.shape

    # ROI抽出
    roi_array = base_data * (mask_data > 0.5)
    
    transform_info = {
        'original_shape': original_shape,
        'affine': affine,
        'mode': mode
    }

    if mode == 'crop_and_pad':
        # 1. Crop
        bbox = get_bbox(mask_data > 0.5) # マスク範囲でクロップ
        (r0, r1), (c0, c1), (s0, s1) = bbox
        cropped_roi = roi_array[r0:r1, c0:c1, s0:s1]
        
        transform_info['bbox'] = bbox
        transform_info['cropped_shape'] = cropped_roi.shape

        # 2. Pad
        padded_roi, pads = pad_to_canvas(cropped_roi, canvas_shape)
        transform_info['pads'] = pads
        
        processed_img = padded_roi
        
    elif mode == 'simple_mask':
        processed_img = roi_array
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Normalize (data_handling.pyのnormalize_intensityに相当する処理)
    # 簡易的に0-1正規化と仮定 (実際の実装に合わせて修正してください)
    if processed_img.max() > 0:
        processed_img = processed_img / processed_img.max()

    return processed_img, transform_info

def restore_heatmap_to_original_space(heatmap, transform_info):
    """
    Grad-CAMのヒートマップ(Canvasサイズ)を元の全脳空間(Originalサイズ)に戻す
    """
    mode = transform_info['mode']
    original_shape = transform_info['original_shape']
    
    if mode == 'simple_mask':
        return heatmap # そのまま

    elif mode == 'crop_and_pad':
        # 1. Un-Pad (パディング部分を除去)
        pads = transform_info['pads']
        # パディング量に基づいてスライスを作成
        slices = []
        for i, (pad_before, pad_after) in enumerate(pads):
            length = heatmap.shape[i]
            # パディングを除去した範囲
            start = pad_before
            end = length - pad_after
            slices.append(slice(start, end))
        
        unpadded_heatmap = heatmap[tuple(slices)]
        
        # サイズ整合性チェック (数値誤差対策)
        target_crop_shape = transform_info['cropped_shape']
        # 必要に応じて微調整リサイズを入れるか、スライスだけで合うはず
        
        # 2. Un-Crop (元の全脳空間の正しい位置に埋め込む)
        restored_volume = np.zeros(original_shape, dtype=np.float32)
        (r0, r1), (c0, c1), (s0, s1) = transform_info['bbox']
        
        # Unpadded heatmapのサイズとbboxサイズが一致するか確認
        # pad_to_canvasの逆操作で厳密に戻す
        d0, d1, d2 = unpadded_heatmap.shape
        # 埋め込み先のサイズに合わせてトリミングまたはパディングが必要な場合の安全策
        # ここでは単純代入できる前提
        try:
            restored_volume[r0:r0+d0, c0:c0+d1, s0:s0+d2] = unpadded_heatmap
        except ValueError as e:
            print(f"Warning during shape restoration: {e}. Attempting resize.")
            # 形状が合わない場合のフェイルセーフなどを検討
            pass
            
        return restored_volume

# ==========================================
# Grad-CAM & Atlas Logic
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
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        # SavedModelの読み込みに失敗した場合、tf.saved_model.loadを試す等の分岐も可能
        raise e

def compute_3d_gradcam(model, img_array, score_mode='increase', target_layer=None):
    replace2linear = ReplaceToLinear()
    gradcam = Gradcam(model, model_modifier=replace2linear, clone=True)
    score = RegressionScore(mode=score_mode)
    
    # (X, Y, Z) -> (1, X, Y, Z, 1)
    input_tensor = np.expand_dims(img_array, axis=0) 
    if input_tensor.ndim == 4:
        input_tensor = np.expand_dims(input_tensor, axis=-1)

    cam = gradcam(score, input_tensor, penultimate_layer=target_layer, seek_penultimate_conv_layer=True)
    heatmap = cam[0] # (X, Y, Z)
    
    # 正規化
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap

def quantize_with_atlas(heatmap_nii, atlas_name='aal'):
    if atlas_name == 'aal':
        dataset = datasets.fetch_atlas_aal(version='SPM12')
    else:
        raise ValueError("Unsupported atlas")

    masker = NiftiLabelsMasker(labels_img=dataset.maps, standardize=False, resampling_target='data', verbose=0)
    
    # 4D化
    data = heatmap_nii.get_fdata()
    img_4d = image.new_img_like(heatmap_nii, np.expand_dims(data, axis=3))
    
    scores = masker.fit_transform(img_4d)
    
    results = []
    for i, label in enumerate(dataset.labels):
        if hasattr(label, 'decode'):
            label = label.decode('utf-8')
        results.append((label, scores[0, i]))
        
    results.sort(key=lambda x: x[1], reverse=True)
    return results