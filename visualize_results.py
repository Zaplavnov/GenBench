import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import argparse

def load_attribution_data(npz_file):
    """
    Загружает данные атрибуции из NPZ файла
    
    Args:
        npz_file: путь к файлу .npz с данными атрибуции
        
    Returns:
        sequence: последовательность ДНК
        attributions: словарь с атрибуциями для разных методов
    """
    data = np.load(npz_file, allow_pickle=True)
    
    # Преобразуем последовательность из массива байтов в строку
    sequence = str(data['sequence'])
    
    # Извлекаем атрибуции для разных методов
    attributions = {
        'Saliency': data['saliency'],
        'Integrated Gradients': data['integrated_gradients'],
        'DeepLift': data['deeplift']
    }
    
    return sequence, attributions

def find_significant_regions(attribution, percentile=95):
    """
    Находит значимые регионы на основе процентиля значений атрибуции
    
    Args:
        attribution: массив значений атрибуции
        percentile: процентиль для определения порога (по умолчанию 95)
        
    Returns:
        significant_indices: индексы значимых позиций
    """
    threshold = np.percentile(attribution, percentile)
    significant_indices = np.where(attribution >= threshold)[0]
    return significant_indices

def create_dna_heatmap(sequence, attributions, title, output_path=None):
    """
    Создает тепловую карту атрибуций для последовательности ДНК
    
    Args:
        sequence: последовательность ДНК
        attributions: словарь с атрибуциями для разных методов
        title: заголовок графика
        output_path: путь для сохранения изображения
    """
    # Создаем матрицу данных для тепловой карты
    methods = list(attributions.keys())
    data_matrix = np.array([attributions[method] for method in methods])
    
    # Нормализуем данные для каждого метода отдельно
    for i in range(len(methods)):
        data_matrix[i] = (data_matrix[i] - data_matrix[i].min()) / (data_matrix[i].max() - data_matrix[i].min())
    
    # Создаем цветовую карту
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#ffffff', '#ffcccc', '#ff9999', '#ff6666', '#ff3333', '#ff0000'])
    
    # Создаем фигуру
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Строим тепловую карту
    sns.heatmap(data_matrix, cmap=cmap, linewidths=0.0, linecolor='white', 
                xticklabels=100, yticklabels=methods, cbar_kws={'label': 'Нормализованная атрибуция'})
    
    # Добавляем заголовок и настраиваем оси
    plt.title(title)
    plt.xlabel('Положение в последовательности')
    
    # Сохраняем или показываем
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def extract_motifs(sequence, significant_indices, window=5):
    """
    Извлекает мотивы из последовательности вокруг значимых позиций
    
    Args:
        sequence: последовательность ДНК
        significant_indices: индексы значимых позиций
        window: размер окна с каждой стороны
        
    Returns:
        motifs: список найденных мотивов
        motif_positions: список позиций найденных мотивов
    """
    motifs = []
    motif_positions = []
    
    for idx in significant_indices:
        start = max(0, idx - window)
        end = min(len(sequence), idx + window + 1)
        
        motif = sequence[start:end]
        motif_pos = (start, end - 1)
        
        motifs.append(motif)
        motif_positions.append(motif_pos)
    
    return motifs, motif_positions

def visualize_significant_regions(sequence, attributions, method_name, output_path=None, percentile=95):
    """
    Визуализирует значимые регионы для конкретного метода атрибуции
    
    Args:
        sequence: последовательность ДНК
        attributions: значения атрибуции
        method_name: название метода атрибуции
        output_path: путь для сохранения изображения
        percentile: процентиль для определения порога (по умолчанию 95)
    """
    # Находим значимые регионы
    significant_indices = find_significant_regions(attributions, percentile=percentile)
    
    # Извлекаем мотивы
    motifs, motif_positions = extract_motifs(sequence, significant_indices, window=5)
    
    # Создаем фигуру
    plt.figure(figsize=(15, 5))
    
    # Отображаем атрибуции
    plt.bar(range(len(attributions)), attributions, alpha=0.5, color='skyblue')
    
    # Выделяем значимые регионы
    threshold = np.percentile(attributions, percentile)
    for idx in significant_indices:
        plt.axvspan(idx-0.5, idx+0.5, color='red', alpha=0.3)
    
    # Добавляем горизонтальную линию для порога
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Порог ({percentile}й процентиль)')
    
    # Настраиваем график
    plt.title(f'Значимые регионы по методу {method_name}')
    plt.xlabel('Положение в последовательности')
    plt.ylabel('Значение атрибуции')
    plt.legend()
    
    # Сохраняем или показываем
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Выводим информацию о найденных мотивах
    print(f"\nЗначимые мотивы для метода {method_name}:")
    for i, (motif, pos) in enumerate(zip(motifs, motif_positions)):
        print(f"Мотив {i+1}: {motif} (позиции {pos[0]}-{pos[1]})")
    
    return motifs, motif_positions

def create_focused_visualization(sequence_name, data_dir, output_dir, percentile=95):
    """
    Создает более детальную визуализацию для последовательности
    
    Args:
        sequence_name: название последовательности
        data_dir: директория с данными интерпретации
        output_dir: директория для сохранения результатов
        percentile: процентиль для определения порога (по умолчанию 95)
    """
    # Загружаем данные атрибуции
    npz_file = os.path.join(data_dir, sequence_name, 'attribution_data.npz')
    
    if not os.path.exists(npz_file):
        print(f"Файл {npz_file} не найден")
        return
    
    # Создаем директорию для результатов
    output_sequence_dir = os.path.join(output_dir, sequence_name)
    os.makedirs(output_sequence_dir, exist_ok=True)
    
    # Загружаем данные
    sequence, attributions = load_attribution_data(npz_file)
    
    # Создаем тепловую карту
    heatmap_path = os.path.join(output_sequence_dir, 'heatmap.png')
    create_dna_heatmap(sequence, attributions, f'Тепловая карта атрибуций для {sequence_name}', heatmap_path)
    
    # Визуализируем значимые регионы для каждого метода
    all_motifs = {}
    for method, attr in attributions.items():
        output_path = os.path.join(output_sequence_dir, f'significant_regions_{method.lower().replace(" ", "_")}.png')
        motifs, positions = visualize_significant_regions(sequence, attr, method, output_path, percentile)
        all_motifs[method] = motifs
    
    return sequence, attributions, all_motifs

def main():
    parser = argparse.ArgumentParser(description="Визуализация результатов интерпретации моделей ДНК")
    parser.add_argument("--input_dir", type=str, default="interpret_results_real_data", 
                        help="Директория с результатами интерпретации")
    parser.add_argument("--output_dir", type=str, default="visualized_results", 
                        help="Директория для сохранения визуализаций")
    parser.add_argument("--percentile", type=float, default=95, 
                        help="Процентиль для определения порога значимости")
    
    args = parser.parse_args()
    
    # Создаем директорию для результатов
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Получаем список последовательностей
    sequences = [d for d in os.listdir(args.input_dir) 
                 if os.path.isdir(os.path.join(args.input_dir, d))]
    
    # Обрабатываем каждую последовательность
    results = {}
    for seq_name in sequences:
        print(f"\nОбработка последовательности: {seq_name}")
        sequence, attributions, motifs = create_focused_visualization(
            seq_name, args.input_dir, args.output_dir, args.percentile)
        
        results[seq_name] = {
            'sequence': sequence,
            'attributions': attributions,
            'motifs': motifs
        }
    
    print(f"\nВизуализация завершена. Результаты сохранены в {args.output_dir}")
    
    return results

if __name__ == "__main__":
    main() 