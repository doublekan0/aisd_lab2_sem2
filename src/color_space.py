import numpy as np


def rgb_to_ycbcr(rgb_array: np.ndarray) -> np.ndarray:
    """
    Преобразование RGB в YCbCr

    Args:
        rgb_array: numpy array формы (H, W, 3) с RGB значениями

    Returns:
        numpy array формы (H, W, 3) с YCbCr значениями
    """
    transform_matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ], dtype=np.float64)

    offset = np.array([0, 128, 128], dtype=np.float64)

    h, w, c = rgb_array.shape
    rgb_flat = rgb_array.reshape(-1, 3).astype(np.float64)
    ycbcr_flat = np.dot(rgb_flat, transform_matrix.T) + offset
    ycbcr = ycbcr_flat.reshape(h, w, 3)

    return np.clip(ycbcr, 0, 255).astype(np.uint8)


def ycbcr_to_rgb(ycbcr_array: np.ndarray) -> np.ndarray:
    """
    Преобразование YCbCr в RGB

    Args:
        ycbcr_array: numpy array формы (H, W, 3) с YCbCr значениями

    Returns:
        numpy array формы (H, W, 3) с RGB значениями
    """
    transform_matrix = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ], dtype=np.float64)

    offset = np.array([0, -128, -128], dtype=np.float64)

    h, w, c = ycbcr_array.shape
    ycbcr_flat = ycbcr_array.reshape(-1, 3).astype(np.float64)
    rgb_flat = np.dot(ycbcr_flat + offset, transform_matrix.T)
    rgb = rgb_flat.reshape(h, w, 3)

    return np.clip(rgb, 0, 255).astype(np.uint8)