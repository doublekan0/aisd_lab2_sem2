import numpy as np
from typing import List, Tuple


def downsample_2x(image_array: np.ndarray) -> np.ndarray:
    """
    Даунсэмплинг с коэффициентом 2 (децимация)

    Args:
        image_array: numpy array формы (H, W) или (H, W, C)

    Returns:
        numpy array с размерами H/2 x W/2
    """
    return image_array[::2, ::2]


def upsample_2x_repeat(image_array: np.ndarray) -> np.ndarray:
    """
    Апсэмплинг дублированием пикселей

    Args:
        image_array: numpy array формы (H, W) или (H, W, C)

    Returns:
        numpy array с размерами H*2 x W*2
    """
    if len(image_array.shape) == 2:
        return np.repeat(np.repeat(image_array, 2, axis=0), 2, axis=1)
    else:
        h, w, c = image_array.shape
        result = np.zeros((h * 2, w * 2, c), dtype=image_array.dtype)
        for i in range(c):
            result[:, :, i] = np.repeat(np.repeat(image_array[:, :, i], 2, axis=0), 2, axis=1)
        return result


def linear_interpolation(x1: float, x2: float, y1: float, y2: float, x: float) -> float:
    """
    Линейная интерполяция для двух точек

    Args:
        x1, x2: узлы интерполяции
        y1, y2: значения в узлах
        x: аргумент интерполянта

    Returns:
        интерполированное значение
    """
    if x1 == x2:
        return y1
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def linear_spline(x_nodes: List[float], y_nodes: List[float], x: float) -> float:
    """
    Линейный сплайн

    Args:
        x_nodes: список узлов интерполяции
        y_nodes: список значений в узлах
        x: аргумент линейного сплайна

    Returns:
        значение линейного сплайна
    """
    if x <= x_nodes[0]:
        return y_nodes[0]
    if x >= x_nodes[-1]:
        return y_nodes[-1]

    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= x <= x_nodes[i + 1]:
            return linear_interpolation(x_nodes[i], x_nodes[i + 1],
                                        y_nodes[i], y_nodes[i + 1], x)
    return y_nodes[-1]


def bilinear_interpolation(points: List[Tuple[float, float]],
                           values: List[float],
                           x: float, y: float) -> float:
    """
    Билинейная интерполяция для четырех точек на плоскости

    Args:
        points: список кортежей [(x1,y1), (x1,y2), (x2,y1), (x2,y2)]
        values: список значений [z11, z12, z21, z22]
        x, y: аргумент интерполянта

    Returns:
        значение интерполянта
    """
    (x1, y1), (x1, y2), (x2, y1), (x2, y2) = points
    z11, z12, z21, z22 = values

    # Интерполяция по x для y1 и y2
    if x2 != x1:
        z_y1 = z11 + (z21 - z11) * (x - x1) / (x2 - x1)
        z_y2 = z12 + (z22 - z12) * (x - x1) / (x2 - x1)
    else:
        z_y1 = z11
        z_y2 = z12

    # Интерполяция по y
    if y2 != y1:
        return z_y1 + (z_y2 - z_y1) * (y - y1) / (y2 - y1)
    else:
        return z_y1


def resize_bilinear(image_array: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
    """
    Изменение размера изображения с билинейной интерполяцией

    Args:
        image_array: numpy array формы (H, W) или (H, W, C)
        new_size: кортеж (new_height, new_width)

    Returns:
        изображение нового размера
    """
    h, w = image_array.shape[:2]
    new_h, new_w = new_size

    if len(image_array.shape) == 3:
        channels = image_array.shape[2]
        result = np.zeros((new_h, new_w, channels), dtype=image_array.dtype)
        for c in range(channels):
            result[:, :, c] = _resize_channel_bilinear(image_array[:, :, c], new_h, new_w)
    else:
        result = _resize_channel_bilinear(image_array, new_h, new_w)

    return result


def _resize_channel_bilinear(channel: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """
    Билинейная интерполяция для одного канала
    """
    h, w = channel.shape
    result = np.zeros((new_h, new_w), dtype=channel.dtype)

    x_scale = (w - 1) / (new_w - 1) if new_w > 1 else 0
    y_scale = (h - 1) / (new_h - 1) if new_h > 1 else 0

    for i in range(new_h):
        for j in range(new_w):
            x = j * x_scale
            y = i * y_scale

            x1 = int(np.floor(x))
            y1 = int(np.floor(y))
            x2 = min(x1 + 1, w - 1)
            y2 = min(y1 + 1, h - 1)

            points = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
            values = [channel[y1, x1], channel[y2, x1],
                      channel[y1, x2], channel[y2, x2]]

            result[i, j] = bilinear_interpolation(points, values, x, y)

    return result