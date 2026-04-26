import numpy as np
from typing import List, Tuple

HUFFMAN_DC_LUMINANCE = {
    0: '00',
    1: '010',
    2: '011',
    3: '100',
    4: '101',
    5: '110',
    6: '1110',
    7: '11110',
    8: '111110',
    9: '1111110',
    10: '11111110',
    11: '111111110'
}

HUFFMAN_AC_LUMINANCE = {
    (0, 0): '1010',
    (0, 1): '00',
    (0, 2): '01',
    (0, 3): '100',
    (0, 4): '1011',
    (0, 5): '11010',
    (0, 6): '1111000',
    (0, 7): '11111000',
    (1, 1): '1100',
    (1, 2): '11011',
    (1, 3): '1111001',
    (1, 4): '111110110',
    (2, 1): '11100',
    (2, 2): '11111001',
    (2, 3): '1111110111',
    (3, 1): '111010',
    (3, 2): '111110111',
    (4, 1): '111011',
    (5, 1): '1111010',
    (6, 1): '11111010',
    (7, 1): '111111000',
    (8, 1): '111111001',
    (9, 1): '111111010',
    (10, 1): '1111110110',
    (15, 0): '11111111001'  # ZRL (Zero Run Length) - 16 нулей
}


def get_category(value: int) -> int:
    """
    Определение категории значения для кодирования Хаффмана

    Args:
        value: значение

    Returns:
        категория (количество бит для представления)
    """
    if value == 0:
        return 0
    return int(np.floor(np.log2(abs(value)))) + 1


def dpcm_encode_dc(dc_coefficients: List[int]) -> List[int]:
    """
    Разностное кодирование DC коэффициентов

    Args:
        dc_coefficients: список DC коэффициентов

    Returns:
        список разностных кодов
    """
    if not dc_coefficients:
        return []

    result = [dc_coefficients[0]]
    for i in range(1, len(dc_coefficients)):
        result.append(dc_coefficients[i] - dc_coefficients[i - 1])

    return result


def dpcm_decode_dc(dpcm_codes: List[int]) -> List[int]:
    """
    Декодирование DPCM для DC коэффициентов

    Args:
        dpcm_codes: список разностных кодов

    Returns:
        список DC коэффициентов
    """
    if not dpcm_codes:
        return []

    result = [dpcm_codes[0]]
    for i in range(1, len(dpcm_codes)):
        result.append(result[-1] + dpcm_codes[i])

    return result


def rle_encode_ac(ac_coefficients: List[int]) -> List[Tuple[int, int]]:
    """
    RLE кодирование AC коэффициентов одного блока

    Args:
        ac_coefficients: список из 63 AC коэффициентов

    Returns:
        список пар (run_length, value)
        EOB кодируется как (0, 0)
    """
    result = []
    zeros_count = 0

    for coeff in ac_coefficients:
        if coeff == 0:
            zeros_count += 1
        else:
            while zeros_count >= 16:
                result.append((15, 0))  # ZRL
                zeros_count -= 16
            result.append((zeros_count, coeff))
            zeros_count = 0

    if zeros_count > 0:
        result.append((0, 0))

    return result


def rle_decode_ac(rle_pairs: List[Tuple[int, int]], total_length: int = 63) -> List[int]:
    """
    Декодирование RLE для AC коэффициентов

    Args:
        rle_pairs: список RLE пар
        total_length: ожидаемая длина последовательности (обычно 63)

    Returns:
        список AC коэффициентов
    """
    result = []

    for run_length, value in rle_pairs:
        if run_length == 0 and value == 0:
            result.extend([0] * (total_length - len(result)))
            break
        else:
            result.extend([0] * run_length)
            result.append(value)

    if len(result) < total_length:
        result.extend([0] * (total_length - len(result)))

    return result[:total_length]


def encode_huffman_dc(dc_diff: int, table: dict = None) -> str:
    """
    Кодирование DC коэффициента с использованием таблиц Хаффмана

    Args:
        dc_diff: разностный DC коэффициент
        table: таблица Хаффмана (по умолчанию для яркости)

    Returns:
        строка бит
    """
    if table is None:
        table = HUFFMAN_DC_LUMINANCE

    category = get_category(dc_diff)
    huff_code = table.get(category, '')

    if dc_diff > 0:
        value_bits = format(dc_diff, f'0{category}b')
    elif dc_diff < 0:
        value_bits = format((1 << category) + dc_diff - 1, f'0{category}b')
    else:
        value_bits = ''

    return huff_code + value_bits


def encode_huffman_ac(run_length: int, value: int, table: dict = None) -> str:
    """
    Кодирование AC коэффициента с использованием таблиц Хаффмана

    Args:
        run_length: количество предшествующих нулей
        value: значение AC коэффициента
        table: таблица Хаффмана (по умолчанию для яркости)

    Returns:
        строка бит
    """
    if table is None:
        table = HUFFMAN_AC_LUMINANCE

    if run_length == 0 and value == 0:
        return table.get((0, 0), '1010')

    category = get_category(value)
    huff_code = table.get((run_length, category), '')

    if value > 0:
        value_bits = format(value, f'0{category}b')
    elif value < 0:
        value_bits = format((1 << category) + value - 1, f'0{category}b')
    else:
        value_bits = ''

    return huff_code + value_bits