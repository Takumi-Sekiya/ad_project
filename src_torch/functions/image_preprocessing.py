import numpy as np

def crop_roi(roi_array):
    """
    ROIの周囲の0埋めの部分を削除する
    """
    if np.all(roi_array == 0):
        return np.zeros((1, 1, 1), dtype=roi_array.dtype)
    indeces = np.array(np.where(roi_array != 0))
    min_coords = indeces.min(axis=1)
    max_coords = indeces.max(axis=1)
    return roi_array[min_coords[0]:max_coords[0]+1,
                     min_coords[1]:max_coords[1]+1,
                     min_coords[2]:max_coords[2]+1]

def pad_to_canvas(cropped_array, canvas_shape):
    """
    指定されたcanvas_shapeの中央に画像を配置する
    """
    canvas = np.zeros(canvas_shape, dtype=cropped_array.dtype)
    start_coords = [(cs - da) // 2 for cs, da in zip(canvas_shape, cropped_array.shape)]
    end_coords = [sc + da for sc, da in zip(start_coords, cropped_array.shape)]
    canvas[start_coords[0]:end_coords[0],
              start_coords[1]:end_coords[1],
              start_coords[2]:end_coords[2]] = cropped_array
    return canvas

def get_non_zero_bounds(roi_array):
    """
    非ゼロ領域（マスク）の最小・最大インデックスを取得する
    """
    if np.all(roi_array == 0):
        return None, None
    indices = np.array(np.where(roi_array != 0))
    # (min_x, min_y, min_z), (max_x, max_y, max_z) の形式で返す
    return indices.min(axis=1), indices.max(axis=1)

def crop_by_ranges(array, ranges):
    """
    指定された固定座標 (ranges) に従って画像を切り出す
    ranges: [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
    ※ max はスライス用に既に +1 された排他的(exclusive)な値であることを前提とする
    """
    return array[ranges[0][0]:ranges[0][1],
                 ranges[1][0]:ranges[1][1],
                 ranges[2][0]:ranges[2][1]]

def normalize_intensity(array):
    """
    画像の輝度値から[0, 1]の範囲に正規化する
    """
    min_val, max_val = array.min(), array.max()
    if max_val - min_val > 0:
        return (array - min_val) / (max_val - min_val)
    return array