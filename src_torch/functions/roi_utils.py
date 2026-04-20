import numpy as np
import nibabel as nib
from scipy.signal import correlate

def load_images(heat_map_path, mask_image_path, threshold=100):
    heat_map = nib.load(heat_map_path).get_fdata().astype(np.float32)
    mask_image = nib.load(mask_image_path).get_fdata().astype(np.float32)

    if mask_image.max() > 1.0:
        mask_image = np.where(mask_image >= threshold, 1.0, 0.0)

    indeces = np.array(np.where(mask_image != 0))
    if indeces.size == 0:
        raise ValueError("Mask image is completely empty (all zeros).")
        
    min_coords = indeces.min(axis=1)
    max_coords = indeces.max(axis=1)
    mask_image = mask_image[min_coords[0]:max_coords[0]+1,
                            min_coords[1]:max_coords[1]+1,
                            min_coords[2]:max_coords[2]+1]
    
    return heat_map, mask_image

def calculate_optimal_shift(heat_map_image, mask_image):
    heat_binary = np.where(heat_map_image > 0, 1.0, 0.0)
    # Use scipy.signal.correlate for efficient cross-correlation
    correlation = correlate(heat_binary, mask_image, mode='valid')
    best_shift = np.unravel_index(np.argmax(correlation), correlation.shape)
    
    return best_shift

def calculate_importance_score(heat_map, mask_image, best_idx):
    dx, dy, dz = best_idx
    sx, sy, sz = mask_image.shape
    heat_patch = heat_map[dx:dx+sx, dy:dy+sy, dz:dz+sz]
    
    binary_heat_patch = np.where(heat_patch >= 0.5, 1.0, 0.0) * mask_image
    heat_patch = heat_patch * binary_heat_patch

    dot = np.sum(heat_patch)
    mask_sum = np.sum(binary_heat_patch)
    return dot / (mask_sum + 1e-8)

def calculate_recall_importance_score(heat_map, mask_image, best_idx, threshold=0.5):
    dx, dy, dz = best_idx
    sx, sy, sz = mask_image.shape
    binary_heat_map = (heat_map > threshold).astype(np.float32)
    heat_patch = binary_heat_map[dx:dx+sx, dy:dy+sy, dz:dz+sz]

    dot = np.sum(heat_patch * mask_image)
    binary_sum = np.sum(binary_heat_map)
    return dot / (binary_sum + 1e-8)

def calculate_threshold_importance_score(heat_map, mask_image, best_idx, threshold=0.7):
    dx, dy, dz = best_idx
    sx, sy, sz = mask_image.shape
    binary_heat_map = (heat_map > threshold).astype(np.float32)
    heat_patch = binary_heat_map[dx:dx+sx, dy:dy+sy, dz:dz+sz]

    dot = np.sum(heat_patch * mask_image)
    binary_sum = np.sum(mask_image)
    return dot / (binary_sum + 1e-8)

def process_subject_roi(args):
    """マルチプロセス用のラッパー関数: 1人の被験者の全サブROIスコアを計算"""
    sub_id, heat_map_path, mask_dir, sub_rois, mask_template, score_type, config_thresholds = args
    importance_scores = {}
    
    for sub_roi in sub_rois:
        mask_image_path = mask_dir / mask_template.format(sub_id=sub_id, sub_roi=sub_roi)
        
        try:
            heat_map_image, mask_image = load_images(heat_map_path, mask_image_path, config_thresholds['mask_binarize'])
            optimal_shift = calculate_optimal_shift(heat_map_image, mask_image)
            
            # スコアタイプの分岐
            if score_type == 'standard':
                score = calculate_importance_score(heat_map_image, mask_image, optimal_shift)
            elif score_type == 'recall':
                score = calculate_recall_importance_score(heat_map_image, mask_image, optimal_shift, config_thresholds['recall_score'])
            elif score_type == 'threshold':
                score = calculate_threshold_importance_score(heat_map_image, mask_image, optimal_shift, config_thresholds['threshold_score'])
            else:
                score = np.nan
                
            importance_scores[sub_roi] = score
        except Exception as e:
            # エラー発生時はNaNを設定（Notebookの動きを再現）
            importance_scores[sub_roi] = np.nan
            
    return sub_id, importance_scores