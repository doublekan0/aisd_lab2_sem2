import numpy as np
from typing import List


def zigzag_scan_square(matrix: np.ndarray) -> List:
    """
    Зигзаг-обход квадратной матрицы NxN

    Args:
        matrix: матрица размера NxN

    Returns:
        список значений в порядке зигзаг-обхода
    """10
    n = matrix.shape[0]
    result = []

    for sum_idx in range(2 * n - 1):
        if sum_idx % 2 == 0:
            # Движение вверх-вправо
            for i in range(min(sum_idx, n - 1), max(-1, sum_idx - n), -1):
                j = sum_idx - i
                if 0 <= i < n and 0 <= j < n:
                    result.append(matrix[i, j])
        else:
            # Движение вниз-влево
            for i in range(max(0, sum_idx - n + 1), min(sum_idx + 1, n)):
                j = sum_idx - i
                if 0 <= i < n and 0 <= j < n:
                    result.append(matrix[i, j])

    return result


def zigzag_scan_rectangular(matrix: np.ndarray) -> List:
    """
    Зигзаг-обход прямоугольной матрицы NxM

    Args:
        matrix: матрица размера NxM

    Returns:
        список значений в порядке зигзаг-обхода
    """
    h, w = matrix.shape
    result = []

    for sum_idx in range(h + w - 1):
        if sum_idx % 2 == 0:
            for i in range(min(sum_idx, h - 1), max(-1, sum_idx - w), -1):
                j = sum_idx - i
                if 0 <= i < h and 0 <= j < w:
                    result.append(matrix[i, j])
        else:
            for i in range(max(0, sum_idx - w + 1), min(sum_idx + 1, h)):
                j = sum_idx - i
                if 0 <= i < h and 0 <= j < w:
                    result.append(matrix[i, j])

    return result


def inverse_zigzag_scan(data: List, n: int) -> np.ndarray:
    """
    Обратный зигзаг-обход для квадратной матрицы

    Args:
        data: список значений
        n: размер матрицы

    Returns:
        матрица NxN
    """
    matrix = np.zeros((n, n), dtype=type(data[0]) if data else np.float64)
    idx = 0

    for sum_idx in range(2 * n - 1):
        if sum_idx % 2 == 0:
            for i in range(min(sum_idx, n - 1), max(-1, sum_idx - n), -1):
                j = sum_idx - i
                if 0 <= i < n and 0 <= j < n and idx < len(data):
                    matrix[i, j] = data[idx]
                    idx += 1
        else:
            for i in range(max(0, sum_idx - n + 1), min(sum_idx + 1, n)):
                j = sum_idx - i
                if 0 <= i < n and 0 <= j < n and idx < len(data):
                    matrix[i, j] = data[idx]
                    idx += 1

    return matrix