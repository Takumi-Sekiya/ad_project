import numpy as np
import nibabel as nib
from scipy.signal import correlate

def crop_by_ranges(array, ranges):
    """絶対座標による切り出し"""
    return array[ranges[0][0]:ranges[0][1],
                 ranges[1][0]:ranges[1][1],
                 ranges[2][0]:ranges[2][1]]

def load_images(heat_map_path, mask_image_path, crop_ranges, threshold=100):
    """
    ヒートマップ（すでにメインROIサイズ）と、
    サブROIマスク（全体サイズからメインROIと同じ座標で切り出し）を読み込む
    """
    heat_map = nib.load(heat_map_path).get_fdata().astype(np.float32)
    mask_image = nib.load(mask_image_path).get_fdata().astype(np.float32)

    if mask_image.max() > 1.0:
        mask_image = np.where(mask_image >= threshold, 1.0, 0.0)

    # メインROIと同じ座標でサブROIを切り出すことでサイズと位置が完全に一致する
    mask_image = crop_by_ranges(mask_image, crop_ranges)
    
    return heat_map, mask_image
"""
def calculate_optimal_shift(heat_map_image, mask_image):
    heat_binary = np.where(heat_map_image > 0, 1.0, 0.0)
    correlation = correlate(heat_binary, mask_image, mode='valid')
    best_shift = np.unravel_index(np.argmax(correlation), correlation.shape)
    return best_shift
"""
def calculate_mean_activation_score(heat_map, mask_image, threshold=0.5):
    masked_heat = heat_map * mask_image
    valid_pixels = (masked_heat > threshold).astype(np.float32)
    
    val_sum = np.sum(masked_heat * valid_pixels)
    count = np.sum(valid_pixels)
    return val_sum / (count + 1e-8)

def calculate_precision_score(heat_map, mask_image, threshold=0.5):
    binary_heat_map = (heat_map > threshold).astype(np.float32)
    intersection = np.sum(binary_heat_map * mask_image)
    predicted_positive = np.sum(binary_heat_map)
    return intersection / (predicted_positive + 1e-8)

def calculate_coverage_score(heat_map, mask_image, threshold=0.7):
    binary_heat_map = (heat_map > threshold).astype(np.float32)
    intersection = np.sum(binary_heat_map * mask_image)
    actual_positive = np.sum(mask_image)
    return intersection / (actual_positive + 1e-8)

def process_subject_roi(args):
    """マルチプロセス用のラッパー関数"""
    sub_id, heat_map_path, mask_dir, sub_rois, mask_template, score_type, config_thresholds, crop_ranges = args
    importance_scores = {}
    
    for sub_roi in sub_rois:
        mask_image_path = mask_dir / mask_template.format(sub_id=sub_id, sub_roi=sub_roi)
        
        try:
            heat_map_image, mask_image = load_images(
                heat_map_path, mask_image_path, crop_ranges, config_thresholds['mask_binarize']
            )
            
            if score_type == 'mean_activation':
                score = calculate_mean_activation_score(heat_map_image, mask_image, config_thresholds['mean_activation'])
            elif score_type == 'precision':
                score = calculate_precision_score(heat_map_image, mask_image, config_thresholds['precision'])
            elif score_type == 'coverage':
                score = calculate_coverage_score(heat_map_image, mask_image, config_thresholds['coverage'])
            else:
                score = np.nan
                
            importance_scores[sub_roi] = score
        except Exception as e:
            importance_scores[sub_roi] = np.nan
            
    # シフト計算を排除したため、3つ目の戻り値はNone固定
    return sub_id, importance_scores, None