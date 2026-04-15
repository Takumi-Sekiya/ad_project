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

def normalize_intensity(array):
    """
    画像の輝度値から[0, 1]の範囲に正規化する
    """
    min_val, max_val = array.min(), array.max()
    if max_val - min_val > 0:
        return (array - min_val) / (max_val - min_val)
    return array