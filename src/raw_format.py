import numpy as np
import json
import struct
from PIL import Image
import os


class RawImage:
    def __init__(self, data: np.ndarray, metadata: dict):
        self.data = data
        self.metadata = metadata

    def save(self, filename: str):
        save_custom_raw(self.data, filename, self.metadata['type'], self.metadata['width'], self.metadata['height'])

    @classmethod
    def load(cls, filename: str) -> 'RawImage':
        data, metadata = load_custom_raw(filename)
        return cls(data, metadata)


def save_custom_raw(data: np.ndarray, filename: str, img_type: str, width: int, height: int):
    with open(filename, 'wb') as f:
        metadata = {'type': img_type, 'width': width, 'height': height, 'channels': 1 if img_type in ['bw', 'gray'] else 3, 'bytes_per_pixel': 1 if img_type in ['bw', 'gray'] else 3}
        metadata_bytes = json.dumps(metadata).encode('utf-8')
        f.write(struct.pack('I', len(metadata_bytes)))
        f.write(metadata_bytes)
        if img_type == 'bw':
            packed = np.packbits(data.flatten() // 255)
            f.write(packed.tobytes())
        else:
            f.write(data.tobytes())


def load_custom_raw(filename: str) -> tuple:
    with open(filename, 'rb') as f:
        header_len = struct.unpack('I', f.read(4))[0]
        metadata = json.loads(f.read(header_len).decode('utf-8'))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        if metadata['type'] == 'bw':
            data = np.unpackbits(data)[:metadata['width'] * metadata['height']]
            data = data.reshape((metadata['height'], metadata['width'])) * 255
            data = data[:, :, np.newaxis]
        else:
            if metadata['channels'] == 1:
                data = data.reshape((metadata['height'], metadata['width'], 1))
            else:
                data = data.reshape((metadata['height'], metadata['width'], 3))
    return data, metadata


def process_image_to_all_formats(img: Image.Image, image_name: str, output_dir: str = 'data/input') -> dict:
    if img.size != (512, 512):
        img = img.resize((512, 512), Image.Resampling.LANCZOS)
    sizes = {}
    img_color = np.array(img)
    color_raw_file = f'{output_dir}/{image_name}_color.raw'
    save_custom_raw(img_color, color_raw_file, 'color', 512, 512)
    img.save(f'{output_dir}/{image_name}_original.png')
    sizes['color_raw'] = os.path.getsize(color_raw_file)
    img_gray = img.convert('L')
    img_gray_array = np.array(img_gray)[:, :, np.newaxis]
    gray_raw_file = f'{output_dir}/{image_name}_gray.raw'
    save_custom_raw(img_gray_array, gray_raw_file, 'gray', 512, 512)
    img_gray.save(f'{output_dir}/{image_name}_gray.png')
    sizes['gray_raw'] = os.path.getsize(gray_raw_file)
    img_bw = img.convert('1', dither=Image.Dither.NONE)
    img_bw_array = (np.array(img_bw) * 255).astype(np.uint8)[:, :, np.newaxis]
    bw_raw_file = f'{output_dir}/{image_name}_bw.raw'
    save_custom_raw(img_bw_array, bw_raw_file, 'bw', 512, 512)
    img_bw.save(f'{output_dir}/{image_name}_bw.png')
    sizes['bw_raw'] = os.path.getsize(bw_raw_file)
    img_bw_dither = img.convert('1', dither=Image.Dither.FLOYDSTEINBERG)
    img_bw_dither_array = (np.array(img_bw_dither) * 255).astype(np.uint8)[:, :, np.newaxis]
    bw_dither_raw_file = f'{output_dir}/{image_name}_bw_dither.raw'
    save_custom_raw(img_bw_dither_array, bw_dither_raw_file, 'bw', 512, 512)
    img_bw_dither.save(f'{output_dir}/{image_name}_bw_dither.png')
    sizes['bw_dither_raw'] = os.path.getsize(bw_dither_raw_file)
    return sizes


def prepare_test_images(input_images: list = None, output_dir: str = 'data/input') -> dict:
    os.makedirs(output_dir, exist_ok=True)
    if input_images is None:
        input_images = ['Lena.png', 'image2.png']
    results = {}
    for idx, img_path in enumerate(input_images):
        print(f"\nОбработка изображения {idx + 1}: {img_path}")
        image_name = 'lena' if idx == 0 else f'image{idx + 1}'
        img = Image.open(img_path)
        print(f"  Загружено: {img.size}, формат: {img.format}")
        results[image_name] = {'original_size': os.path.getsize(img_path), 'original_format': img.format}
        sizes = process_image_to_all_formats(img, image_name, output_dir)
        results[image_name]['raw_sizes'] = sizes
        print(f"\n  Статистика для {image_name}:")
        print(f"    Исходный размер: {results[image_name]['original_size']:,} байт")
        print(f"    Цветное RAW: {sizes['color_raw']:,} байт (сжатие: {results[image_name]['original_size'] / sizes['color_raw']:.2f}x)")
        print(f"    Серое RAW: {sizes['gray_raw']:,} байт (сжатие: {results[image_name]['original_size'] / sizes['gray_raw']:.2f}x)")
        print(f"    Ч/Б RAW без дизеринга: {sizes['bw_raw']:,} байт (сжатие: {results[image_name]['original_size'] / sizes['bw_raw']:.2f}x)")
        print(f"    Ч/Б RAW с дизерингом: {sizes['bw_dither_raw']:,} байт")
    return results


def print_comparison_table(results: dict):
    print("\n" + "=" * 80)
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА РАЗМЕРОВ ФАЙЛОВ")
    print("=" * 80)
    for image_name, data in results.items():
        print(f"\n{image_name.upper()} ({data['original_format']}):")
        print("-" * 60)
        print(f"  Исходный файл: {data['original_size']:,} байт")
        print(f"  RAW цветное: {data['raw_sizes']['color_raw']:,} байт (байт/пиксель: {data['raw_sizes']['color_raw'] / (512 * 512):.2f})")
        print(f"  RAW серое: {data['raw_sizes']['gray_raw']:,} байт (байт/пиксель: {data['raw_sizes']['gray_raw'] / (512 * 512):.2f})")
        print(f"  RAW Ч/Б: {data['raw_sizes']['bw_raw']:,} байт (байт/пиксель: {data['raw_sizes']['bw_raw'] / (512 * 512):.2f})")
        print(f"  RAW Ч/Б с дизерингом: {data['raw_sizes']['bw_dither_raw']:,} байт")
        print(f"\n  Коэффициенты сжатия относительно исходного:")
        print(f"    -> Цветное RAW: {data['original_size'] / data['raw_sizes']['color_raw']:.2f}x")
        print(f"    -> Серое RAW: {data['original_size'] / data['raw_sizes']['gray_raw']:.2f}x")
        print(f"    -> Ч/Б RAW: {data['original_size'] / data['raw_sizes']['bw_raw']:.2f}x")