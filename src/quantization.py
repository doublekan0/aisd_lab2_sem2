import numpy as np


def get_standard_q_table() -> np.ndarray:
    """
    Стандартная таблица квантования JPEG для яркости

    Returns:
        таблица квантования 8x8
    """
    return np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float64)


def get_chrominance_q_table() -> np.ndarray:
    """
    Стандартная таблица квантования JPEG для цветности

    Returns:
        таблица квантования 8x8
    """
    return np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ], dtype=np.float64)


def adapt_quantization_table(base_table: np.ndarray, quality: int) -> np.ndarray:
    """
    Адаптация таблицы квантования под уровень качества

    Формулы:
        Quality ∈ [1, 50): S = 5000 / Quality
        Quality ∈ [50, 100]: S = 200 - 2 * Quality
        q' = ceil((q * S) / 100)

    Args:
        base_table: базовая таблица квантования
        quality: уровень качества (1-100)

    Returns:
        адаптированная таблица квантования
    """
    quality = max(1, min(100, quality))

    if quality < 50:
        S = 5000 / quality
    else:
        S = 200 - 2 * quality

    adapted = np.ceil((base_table * S) / 100)
    adapted = np.clip(adapted, 1, 255)

    return adapted.astype(np.uint8)


def quantize(dct_block: np.ndarray, q_table: np.ndarray) -> np.ndarray:
    """
    Квантование коэффициентов DCT
    c' = round(c / q)

    Args:
        dct_block: матрица коэффициентов DCT
        q_table: матрица коэффициентов квантования

    Returns:
        матрица квантованных коэффициентов
    """
    return np.round(dct_block / q_table).astype(np.int32)


def dequantize(quantized_block: np.ndarray, q_table: np.ndarray) -> np.ndarray:
    """
    Деквантование (нормализация) коэффициентов
    c = c' * q

    Args:
        quantized_block: матрица квантованных коэффициентов
        q_table: матрица коэффициентов квантования

    Returns:
        матрица нормализованных коэффициентов
    """
    return (quantized_block * q_table).astype(np.float64)