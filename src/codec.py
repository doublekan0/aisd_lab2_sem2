import numpy as np
import json
import struct
from typing import Dict, List, Tuple

from .color_space import rgb_to_ycbcr, ycbcr_to_rgb
from .dct import dct_2d_matrix, idct_2d_matrix, split_into_blocks_8x8
from .quantization import get_standard_q_table, get_chrominance_q_table, adapt_quantization_table, quantize, dequantize
from .zigzag import zigzag_scan_square, inverse_zigzag_scan


class JPEGCodec:
    """
    JPEG-подобный кодек

    Attributes:
        quality: уровень качества (1-100)
        color_space: цветовое пространство ('RGB', 'YCbCr', 'GRAY')
        q_table_y: таблица квантования для яркости
        q_table_c: таблица квантования для цветности
    """

    def __init__(self, quality: int = 50, color_space: str = 'YCbCr'):
        """
        Инициализация кодека

        Args:
            quality: уровень качества (1-100)
            color_space: цветовое пространство
        """
        self.quality = max(1, min(100, quality))
        self.color_space = color_space

        # Создаем таблицы квантования
        base_y = get_standard_q_table()
        base_c = get_chrominance_q_table()

        self.q_table_y = adapt_quantization_table(base_y, self.quality)
        self.q_table_c = adapt_quantization_table(base_c, self.quality)

    def compress(self, image_array: np.ndarray) -> Dict:
        """
        Сжатие изображения

        Args:
            image_array: numpy array с изображением

        Returns:
            словарь со сжатыми данными
        """
        h, w = image_array.shape[:2]

        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            if self.color_space == 'YCbCr':
                working_array = rgb_to_ycbcr(image_array)
            else:
                working_array = image_array
            channels = 3
        else:
            working_array = image_array
            channels = 1

        blocks, _ = split_into_blocks_8x8(working_array)

        compressed_blocks = []

        for block, ch in blocks:
            dct_block = dct_2d_matrix(block - 128)

            q_table = self.q_table_y if ch == 0 else self.q_table_c
            quantized = quantize(dct_block, q_table)

            zigzag = zigzag_scan_square(quantized)

            compressed_blocks.append({
                'channel': int(ch),
                'dc': int(zigzag[0]),
                'ac': [int(x) for x in zigzag[1:]]
            })

        return {
            'width': w,
            'height': h,
            'channels': channels,
            'color_space': self.color_space,
            'quality': self.quality,
            'blocks': compressed_blocks
        }

    def decompress(self, compressed_data: Dict) -> np.ndarray:
        """
        Декомпрессия изображения

        Args:
            compressed_data: словарь со сжатыми данными

        Returns:
            восстановленное изображение
        """
        width = compressed_data['width']
        height = compressed_data['height']
        channels = compressed_data['channels']
        quality = compressed_data['quality']

        q_table_y = adapt_quantization_table(get_standard_q_table(), quality)
        q_table_c = adapt_quantization_table(get_chrominance_q_table(), quality)

        blocks_h = (height + 7) // 8
        blocks_w = (width + 7) // 8

        if channels == 1:
            restored = np.zeros((blocks_h * 8, blocks_w * 8), dtype=np.float64)
        else:
            restored = np.zeros((blocks_h * 8, blocks_w * 8, channels), dtype=np.float64)

        block_idx = 0
        for ch in range(channels):
            q_table = q_table_y if ch == 0 else q_table_c

            for row in range(blocks_h):
                for col in range(blocks_w):
                    if block_idx < len(compressed_data['blocks']):
                        block_data = compressed_data['blocks'][block_idx]

                        zigzag = [block_data['dc']] + block_data['ac']

                        quantized = inverse_zigzag_scan(zigzag, 8)

                        dct_block = dequantize(quantized, q_table)

                        block = idct_2d_matrix(dct_block) + 128

                        if channels == 1:
                            restored[row * 8:(row + 1) * 8, col * 8:(col + 1) * 8] = block
                        else:
                            restored[row * 8:(row + 1) * 8, col * 8:(col + 1) * 8, ch] = block

                        block_idx += 1

        if channels == 1:
            restored = restored[:height, :width]
        else:
            restored = restored[:height, :width, :]

        restored = np.clip(restored, 0, 255).astype(np.uint8)

        if channels == 3 and compressed_data['color_space'] == 'YCbCr':
            restored = ycbcr_to_rgb(restored)

        return restored

    def save(self, compressed_data: Dict, filename: str):
        """
        Сохранение сжатых данных в файл

        Args:
            compressed_data: словарь со сжатыми данными
            filename: имя файла
        """
        with open(filename, 'wb') as f:
            json_data = json.dumps(compressed_data).encode('utf-8')
            f.write(struct.pack('I', len(json_data)))
            f.write(json_data)

    @classmethod
    def load(cls, filename: str) -> Dict:
        """
        Загрузка сжатых данных из файла

        Args:
            filename: имя файла

        Returns:
            словарь со сжатыми данными
        """
        with open(filename, 'rb') as f:
            data_len = struct.unpack('I', f.read(4))[0]
            return json.loads(f.read(data_len).decode('utf-8'))