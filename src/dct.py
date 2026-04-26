import numpy as np
from typing import List, Tuple


def create_dct_matrix(N: int) -> np.ndarray:
    """
    Создание матрицы DCT размера NxN

    Args:
        N: размер матрицы

    Returns:
        матрица DCT размера NxN
    """
    C = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if j == 0:
                C[i, j] = 1 / np.sqrt(N)
            else:
                C[i, j] = np.sqrt(2 / N) * np.cos((2 * i + 1) * j * np.pi / (2 * N))
    return C


def dct_2d_matrix(block: np.ndarray) -> np.ndarray:
    """
    DCT через матричное умножение: S = C * block * C^T
    Временная сложность: O(N^3)

    Args:
        block: блок изображения размера NxM

    Returns:
        матрица коэффициентов DCT
    """
    h, w = block.shape
    C_h = create_dct_matrix(h)
    C_w = create_dct_matrix(w)
    return C_h @ block @ C_w.T


def idct_2d_matrix(dct_block: np.ndarray) -> np.ndarray:
    """
    Обратное DCT через матричное умножение

    Args:
        dct_block: матрица коэффициентов DCT

    Returns:
        восстановленный блок изображения
    """
    h, w = dct_block.shape
    C_h = create_dct_matrix(h)
    C_w = create_dct_matrix(w)
    return C_h.T @ dct_block @ C_w


def dct_2d_primitive(block: np.ndarray) -> np.ndarray:
    """
    Примитивное вычисление DCT для блока произвольного размера
    Временная сложность: O(N^2 * M^2)

    Args:
        block: блок изображения размера NxM

    Returns:
        матрица коэффициентов DCT
    """
    h, w = block.shape
    result = np.zeros((h, w), dtype=np.float64)

    for u in range(h):
        for v in range(w):
            sum_val = 0.0
            for x in range(h):
                for y in range(w):
                    sum_val += block[x, y] * \
                               np.cos((2 * x + 1) * u * np.pi / (2 * h)) * \
                               np.cos((2 * y + 1) * v * np.pi / (2 * w))

            cu = 1 / np.sqrt(2) if u == 0 else 1
            cv = 1 / np.sqrt(2) if v == 0 else 1
            result[u, v] = (2 / np.sqrt(h * w)) * cu * cv * sum_val

    return result


def idct_2d_primitive(dct_block: np.ndarray) -> np.ndarray:
    """
    Примитивное обратное DCT

    Args:
        dct_block: матрица коэффициентов DCT

    Returns:
        восстановленный блок изображения
    """
    h, w = dct_block.shape
    result = np.zeros((h, w), dtype=np.float64)

    for x in range(h):
        for y in range(w):
            sum_val = 0.0
            for u in range(h):
                for v in range(w):
                    cu = 1 / np.sqrt(2) if u == 0 else 1
                    cv = 1 / np.sqrt(2) if v == 0 else 1
                    sum_val += cu * cv * dct_block[u, v] * \
                               np.cos((2 * x + 1) * u * np.pi / (2 * h)) * \
                               np.cos((2 * y + 1) * v * np.pi / (2 * w))
            result[x, y] = (2 / np.sqrt(h * w)) * sum_val

    return result


def split_into_blocks_8x8(image_array: np.ndarray) -> Tuple[List, Tuple[int, int]]:
    """
    Разбиение изображения на блоки 8x8

    Args:
        image_array: numpy array формы (H, W) или (H, W, C)

    Returns:
        tuple: (список блоков с каналами, оригинальный размер)
    """
    h, w = image_array.shape[:2]

    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8

    if len(image_array.shape) == 3:
        padded = np.pad(image_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        channels = image_array.shape[2]
        blocks = []

        for c in range(channels):
            for i in range(0, padded.shape[0], 8):
                for j in range(0, padded.shape[1], 8):
                    block = padded[i:i + 8, j:j + 8, c].astype(np.float64)
                    blocks.append((block, c))
    else:
        padded = np.pad(image_array, ((0, pad_h), (0, pad_w)), mode='edge')
        blocks = []

        for i in range(0, padded.shape[0], 8):
            for j in range(0, padded.shape[1], 8):
                block = padded[i:i + 8, j:j + 8].astype(np.float64)
                blocks.append((block, 0))

    return blocks, (h, w)