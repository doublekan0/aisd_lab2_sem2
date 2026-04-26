import os
import sys
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.raw_format import prepare_test_images, print_comparison_table
from src.color_space import rgb_to_ycbcr, ycbcr_to_rgb
from src.sampling import downsample_2x, upsample_2x_repeat
from src.sampling import linear_interpolation, bilinear_interpolation, linear_spline
from src.dct import dct_2d_matrix, idct_2d_matrix
from src.quantization import get_standard_q_table, quantize
from src.zigzag import zigzag_scan_square, zigzag_scan_rectangular, inverse_zigzag_scan
from src.encoding import dpcm_encode_dc, rle_encode_ac, encode_huffman_dc, encode_huffman_ac
from src.codec import JPEGCodec


def create_directories():
    dirs = ['data/input', 'data/output', 'data/temp', 'results/graphs',
            'results/compressed', 'results/compressed/by_quality']
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def task1_prepare_data():
    print_section("ЗАДАНИЕ 1.1: Подготовка тестовых данных")
    input_images = ['Lena.png', 'image2.png']
    print(f"\nВходные изображения:")
    for i, img in enumerate(input_images, 1):
        print(f"  {i}. {img}")
    results = prepare_test_images(input_images, 'data/input')
    print_comparison_table(results)
    print("\nТестовые данные подготовлены для двух изображений!")
    return results


def task1_color_spaces():
    print_section("ЗАДАНИЕ 1.2: Цветовые пространства RGB - YCbCr")
    test_rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    ycbcr = rgb_to_ycbcr(test_rgb)
    rgb_restored = ycbcr_to_rgb(ycbcr)
    diff = np.max(np.abs(test_rgb.astype(np.int16) - rgb_restored.astype(np.int16)))
    print(f"\nМаксимальная разница после RGB -> YCbCr -> RGB: {diff}")
    print(f"Преобразования работают корректно: {'Да' if diff <= 2 else 'Нет'}")
    return diff <= 2


def task1_sampling():
    print_section("ЗАДАНИЕ 1.3: Downsampling/Upsampling/Интерполяция")
    result_linear = linear_interpolation(0, 10, 0, 100, 5)
    print(f"\nЛинейная интерполяция: x=5 между (0,0) и (10,100) = {result_linear}")
    x_nodes = [0, 10, 20, 30]
    y_nodes = [0, 100, 50, 150]
    result_spline = linear_spline(x_nodes, y_nodes, 15)
    print(f"Линейный сплайн: x=15 -> {result_spline}")
    points = [(0, 0), (0, 1), (1, 0), (1, 1)]
    values = [10, 20, 30, 40]
    result_bilinear = bilinear_interpolation(points, values, 0.5, 0.5)
    print(f"Билинейная интерполяция: (0.5, 0.5) -> {result_bilinear}")
    test_img = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120], [130, 140, 150, 160]])
    downsampled = downsample_2x(test_img)
    print(f"\nДаунсэмплинг 4x4 -> 2x2:\n{downsampled}")
    upsampled = upsample_2x_repeat(downsampled)
    print(f"\nАпсэмплинг 2x2 -> 4x4 (первые 2 строки):\n{upsampled[:2]}")


def task1_dct():
    print_section("ЗАДАНИЕ 1.4: Дискретное косинусное преобразование")
    test_block = np.random.randint(0, 256, (8, 8)).astype(np.float64) - 128
    start = time.time()
    dct_result = dct_2d_matrix(test_block)
    time_dct = time.time() - start
    print(f"\nВремя DCT (8x8): {time_dct * 1000:.3f} мс")
    start = time.time()
    idct_result = idct_2d_matrix(dct_result)
    time_idct = time.time() - start
    print(f"Время IDCT (8x8): {time_idct * 1000:.3f} мс")
    restored = idct_result + 128
    diff = np.max(np.abs(test_block + 128 - restored))
    print(f"Максимальная разница после DCT -> IDCT: {diff:.10f}")
    print(f"Преобразование обратимо: {'Да' if diff < 0.1 else 'Нет'}")
    q_table = get_standard_q_table()
    quantized = quantize(dct_result, q_table)
    non_zero = np.count_nonzero(quantized)
    print(f"\nПосле квантования: {non_zero} ненулевых коэффициентов из 64 ({non_zero / 64 * 100:.1f}%)")
    print(f"\nРекомендуемый тип данных для коэффициентов DCT: float64")


def task2_zigzag():
    print_section("ЗАДАНИЕ 2.1: Зигзаг-обход")
    test_square = np.arange(16).reshape(4, 4)
    print("\nКвадратная матрица 4x4:")
    print(test_square)
    zigzag = zigzag_scan_square(test_square)
    print(f"\nЗигзаг-обход: {zigzag}")
    restored = inverse_zigzag_scan(zigzag, 4)
    print("\nВосстановленная матрица:")
    print(restored.astype(int))
    test_rect = np.arange(12).reshape(3, 4)
    print(f"\nПрямоугольная матрица 3x4:")
    print(test_rect)
    rect_zigzag = zigzag_scan_rectangular(test_rect)
    print(f"Зигзаг-обход: {rect_zigzag}")


def task2_encoding():
    print_section("ЗАДАНИЕ 2.2: Разностное кодирование и RLE")
    dc = [150, 155, 148, 160, 158]
    encoded = dpcm_encode_dc(dc)
    print("\nDPCM кодирование DC коэффициентов:")
    print(f"  Исходные: {dc}")
    print(f"  Разностные коды: {encoded}")
    ac = [5, 0, 0, -2, 0, 0, 0, 1] + [0] * 55
    rle = rle_encode_ac(ac[:63])
    print("\nRLE кодирование AC коэффициентов:")
    print(f"  Исходные (первые 8): {ac[:8]}")
    print(f"  RLE пары: {rle}")
    print("\nКодирование Хаффмана:")
    dc_encoded = encode_huffman_dc(5)
    ac_encoded = encode_huffman_ac(0, 2)
    print(f"  DC(5): {dc_encoded}")
    print(f"  AC(0,2): {ac_encoded}")
    print(f"  EOB: {encode_huffman_ac(0, 0)}")


def task2_quality_test():
    print_section("ЗАДАНИЕ 2.3: Анализ качества сжатия для всех тестовых изображений")
    test_images = [
        {'name': 'Lena_color', 'path': 'data/input/lena_original.png', 'mode': 'RGB', 'display': 'Lena (цветное)'},
        {'name': 'Lena_gray', 'path': 'data/input/lena_gray.png', 'mode': 'L', 'display': 'Lena (серое)'},
        {'name': 'Lena_bw', 'path': 'data/input/lena_bw.png', 'mode': '1', 'display': 'Lena (Ч/Б)'},
        {'name': 'Lena_bw_dither', 'path': 'data/input/lena_bw_dither.png', 'mode': '1', 'display': 'Lena (Ч/Б с дизерингом)'},
        {'name': 'Image2_color', 'path': 'data/input/image2_original.png', 'mode': 'RGB', 'display': 'Image2 (цветное)'},
        {'name': 'Image2_gray', 'path': 'data/input/image2_gray.png', 'mode': 'L', 'display': 'Image2 (серое)'},
        {'name': 'Image2_bw', 'path': 'data/input/image2_bw.png', 'mode': '1', 'display': 'Image2 (Ч/Б)'},
        {'name': 'Image2_bw_dither', 'path': 'data/input/image2_bw_dither.png', 'mode': '1', 'display': 'Image2 (Ч/Б с дизерингом)'}
    ]
    qualities = list(range(10, 101, 10))
    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    axes = axes.flatten()
    all_results = {}
    os.makedirs('data/temp', exist_ok=True)

    for idx, img_info in enumerate(test_images):
        print(f"\nОбработка: {img_info['display']}")
        print("-" * 55)
        try:
            img = Image.open(img_info['path'])
            if img_info['mode'] == '1':
                img = img.convert('L')
            img_array = np.array(img)
        except Exception:
            print(f"  Файл {img_info['path']} не найден, пропускаю...")
            continue
        sizes = []
        psnr_values = []
        print(f"  {'Качество':<10} | {'Размер (байт)':<15} | {'PSNR (dB)':<10}")
        print(f"  " + "-" * 40)
        for q in qualities:
            codec = JPEGCodec(quality=q, color_space='YCbCr' if img_info['mode'] == 'RGB' else 'GRAY')
            compressed = codec.compress(img_array)
            temp_file = f'data/temp/{img_info["name"]}_q{q}.jpgc'
            codec.save(compressed, temp_file)
            size = os.path.getsize(temp_file)
            sizes.append(size)
            restored = codec.decompress(compressed)
            mse = np.mean((img_array.astype(np.float64) - restored.astype(np.float64)) ** 2)
            psnr = float('inf') if mse == 0 else 20 * np.log10(255 / np.sqrt(mse))
            psnr_values.append(psnr)
            print(f"  {q:<10} | {size:<15,d} | {psnr:<10.2f}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
        all_results[img_info['display']] = {'qualities': qualities, 'sizes': sizes, 'psnr': psnr_values}
        if idx < len(axes):
            ax1 = axes[idx]
            ax2 = ax1.twinx()
            ax1.plot(qualities, [s / 1024 for s in sizes], 'b-o', linewidth=2, markersize=7, label='Размер (КБ)')
            ax2.plot(qualities, psnr_values, 'r-s', linewidth=2, markersize=7, label='PSNR (dB)')
            ax1.set_xlabel('Качество', fontsize=10)
            ax1.set_ylabel('Размер (КБ)', color='b', fontsize=10)
            ax2.set_ylabel('PSNR (dB)', color='r', fontsize=10)
            ax1.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='r')
            ax1.set_title(img_info['display'], fontsize=11, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            lines = ax1.get_lines() + ax2.get_lines()
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left', fontsize=9)

    for idx in range(len(test_images), len(axes)):
        fig.delaxes(axes[idx])
    plt.suptitle('Зависимость размера сжатого файла и PSNR от качества сжатия', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/graphs/all_images_quality_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nОбщий график сохранен: results/graphs/all_images_quality_analysis.png")
    for name, data in all_results.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(data['qualities'], [s / 1024 for s in data['sizes']], 'b-o', linewidth=2, markersize=7)
        ax1.set_xlabel('Качество', fontsize=11)
        ax1.set_ylabel('Размер (КБ)', fontsize=11)
        ax1.set_title(f'Размер сжатого файла\n{name}', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax2.plot(data['qualities'], data['psnr'], 'r-o', linewidth=2, markersize=7)
        ax2.set_xlabel('Качество', fontsize=11)
        ax2.set_ylabel('PSNR (dB)', fontsize=11)
        ax2.set_title(f'Качество восстановления\n{name}', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        plt.tight_layout()
        plt.savefig(f'results/graphs/{safe_name}_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    print(f"Индивидуальные графики сохранены в: results/graphs/")
    return all_results


def task2_full_test():
    print_section("ЗАДАНИЕ 2.4: Декомпрессия всех тестовых изображений")
    os.makedirs('results/compressed/by_quality', exist_ok=True)
    test_images = [
        {'name': 'lena_color', 'path': 'data/input/lena_original.png', 'mode': 'RGB', 'display': 'Lena (цветное)'},
        {'name': 'lena_gray', 'path': 'data/input/lena_gray.png', 'mode': 'L', 'display': 'Lena (серое)'},
        {'name': 'lena_bw', 'path': 'data/input/lena_bw.png', 'mode': '1', 'display': 'Lena (Ч/Б)'},
        {'name': 'lena_bw_dither', 'path': 'data/input/lena_bw_dither.png', 'mode': '1', 'display': 'Lena (Ч/Б с дизерингом)'},
        {'name': 'image2_color', 'path': 'data/input/image2_original.png', 'mode': 'RGB', 'display': 'Image2 (цветное)'},
        {'name': 'image2_gray', 'path': 'data/input/image2_gray.png', 'mode': 'L', 'display': 'Image2 (серое)'},
        {'name': 'image2_bw', 'path': 'data/input/image2_bw.png', 'mode': '1', 'display': 'Image2 (Ч/Б)'},
        {'name': 'image2_bw_dither', 'path': 'data/input/image2_bw_dither.png', 'mode': '1', 'display': 'Image2 (Ч/Б с дизерингом)'}
    ]
    qualities = [10, 25, 50, 75, 90]
    print("\nСжатие и восстановление изображений...")
    print("=" * 85)
    all_compressed_sizes = {}
    all_psnr = {}

    for img_info in test_images:
        print(f"\n{img_info['display']}:")
        print("-" * 85)
        try:
            img = Image.open(img_info['path'])
            if img_info['mode'] == '1':
                img = img.convert('L')
            img_array = np.array(img)
            print(f"  Исходный размер: {img_array.shape}, {img_array.nbytes:,} байт")
        except Exception:
            print(f"  Файл {img_info['path']} не найден, пропускаю...")
            continue
        compressed_sizes = []
        psnr_values = []
        print(f"\n  {'Качество':<8} | {'Сжатый размер':<13} | {'Сжатие':<8} | {'PSNR (dB)':<10} | {'Восст. файл'}")
        print(f"  " + "-" * 75)
        for q in qualities:
            codec = JPEGCodec(quality=q, color_space='YCbCr' if img_info['mode'] == 'RGB' else 'GRAY')
            compressed = codec.compress(img_array)
            compressed_file = f'results/compressed/by_quality/{img_info["name"]}_q{q}.jpgc'
            codec.save(compressed, compressed_file)
            compressed_size = os.path.getsize(compressed_file)
            compressed_sizes.append(compressed_size)
            loaded = JPEGCodec.load(compressed_file)
            restored = codec.decompress(loaded)
            restored_file = f'results/compressed/by_quality/{img_info["name"]}_q{q}_restored.png'
            if len(restored.shape) == 2:
                Image.fromarray(restored.astype(np.uint8)).save(restored_file)
            else:
                Image.fromarray(restored.astype(np.uint8)).save(restored_file)
            mse = np.mean((img_array.astype(np.float64) - restored.astype(np.float64)) ** 2)
            psnr = float('inf') if mse == 0 else 20 * np.log10(255 / np.sqrt(mse))
            psnr_values.append(psnr)
            compression_ratio = img_array.nbytes / compressed_size if compressed_size > 0 else 0
            print(f"  {q:<8} | {compressed_size:>10,d} б | {compression_ratio:>5.2f}x | {psnr:>10.2f} | {os.path.basename(restored_file)}")
        all_compressed_sizes[img_info['display']] = compressed_sizes
        all_psnr[img_info['display']] = psnr_values

    print("\n" + "=" * 85)
    print("СВОДНАЯ ТАБЛИЦА: РАЗМЕРЫ СЖАТЫХ ФАЙЛОВ (байт)")
    print("=" * 85)
    header = f"{'Изображение':<28} | " + " | ".join([f"Q{q:3d}" for q in qualities])
    print(header)
    print("-" * len(header))
    for name, sizes in all_compressed_sizes.items():
        print(f"{name:<28} | " + " | ".join([f"{s:7,d}" for s in sizes]))

    print("\n" + "=" * 85)
    print("СВОДНАЯ ТАБЛИЦА: PSNR (dB)")
    print("=" * 85)
    header = f"{'Изображение':<28} | " + " | ".join([f"Q{q:3d}" for q in qualities])
    print(header)
    print("-" * len(header))
    for name, psnr_list in all_psnr.items():
        print(f"{name:<28} | " + " | ".join([f"{p:7.2f}" if p != float('inf') else "    inf" for p in psnr_list]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_compressed_sizes)))
    for idx, (name, sizes) in enumerate(all_compressed_sizes.items()):
        ax1.plot(qualities, [s / 1024 for s in sizes], marker=markers[idx % len(markers)], color=colors[idx], linewidth=2, markersize=8, label=name[:20])
    ax1.set_xlabel('Качество', fontsize=12)
    ax1.set_ylabel('Размер сжатого файла (КБ)', fontsize=12)
    ax1.set_title('Размер сжатых файлов при разном качестве', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    for idx, (name, psnr_list) in enumerate(all_psnr.items()):
        psnr_plot = [p if p != float('inf') else 100 for p in psnr_list]
        ax2.plot(qualities, psnr_plot, marker=markers[idx % len(markers)], color=colors[idx], linewidth=2, markersize=8, label=name[:20])
    ax2.set_xlabel('Качество', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title('Качество восстановления при разном качестве сжатия', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/graphs/compressed_analysis_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 85)
    print("ВСЕ ИЗОБРАЖЕНИЯ ОБРАБОТАНЫ!")
    print("=" * 85)
    print(f"\nРезультаты сохранены:")
    print(f"  results/compressed/by_quality/")
    print(f"  results/graphs/")
    return all_compressed_sizes, all_psnr


def main():
    print("\n" + "=" * 70)
    print("  ЛАБОРАТОРНАЯ РАБОТА 2")
    print("=" * 70)
    create_directories()
    task1_prepare_data()
    task1_color_spaces()
    task1_sampling()
    task1_dct()
    task2_zigzag()
    task2_encoding()
    task2_quality_test()
    task2_full_test()
    print_section("ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ УСПЕШНО!")
    print("\nРезультаты сохранены в директориях:")
    print("  data/input/          : тестовые изображения в RAW формате")
    print("  results/graphs/      : графики анализа качества")
    print("  results/compressed/  : сжатые файлы и восстановленные изображения")
    print("=" * 70)


if __name__ == "__main__":
    main()