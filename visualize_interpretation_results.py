#!/usr/bin/env python

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from pathlib import Path
import glob
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr

def load_attribution_data(results_dir):
    """
    Загружает данные атрибуций из директории результатов
    
    Args:
        results_dir: путь к директории с результатами
        
    Returns:
        dict: словарь с данными атрибуций
    """
    # Ищем все файлы .npz в директории и поддиректориях
    data_files = glob.glob(os.path.join(results_dir, "**", "attribution_data.npz"), recursive=True)
    
    if not data_files:
        raise FileNotFoundError(f"В директории {results_dir} не найдены файлы с данными атрибуций")
    
    # Загружаем данные из каждого файла
    all_data = []
    for data_file in data_files:
        try:
            data = np.load(data_file, allow_pickle=True)
            all_data.append({
                'file': data_file,
                'sequence': str(data['sequence']),
                'saliency': data['saliency'],
                'integrated_gradients': data['integrated_gradients'],
                'deeplift': data['deeplift']
            })
        except Exception as e:
            print(f"Ошибка при загрузке {data_file}: {e}")
    
    return all_data

def visualize_sequence_with_attributions(sequence, attributions, methods, title, output_path=None):
    """
    Визуализирует последовательность ДНК с атрибуциями от разных методов
    
    Args:
        sequence: последовательность ДНК
        attributions: словарь {метод: значения атрибуций}
        methods: список методов для отображения
        title: заголовок графика
        output_path: путь для сохранения изображения
    """
    n_methods = len(methods)
    
    # Создаем фигуру
    fig = plt.figure(figsize=(15, 3 * n_methods))
    gs = GridSpec(n_methods, 1, figure=fig)
    
    # Цветовая карта для нуклеотидов
    nuc_colors = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red', 'N': 'gray'}
    
    # Нормализация данных для сравнимости
    scaler = MinMaxScaler()
    
    # Отображаем каждый метод
    for i, method in enumerate(methods):
        ax = fig.add_subplot(gs[i, 0])
        
        # Нормализуем атрибуции
        normalized_attr = scaler.fit_transform(attributions[method].reshape(-1, 1)).reshape(-1)
        
        # Отображаем атрибуции как столбчатую диаграмму
        bars = ax.bar(range(len(normalized_attr)), normalized_attr, alpha=0.7)
        
        # Добавляем нуклеотиды, если последовательность не слишком длинная
        if len(sequence) <= 100:
            for j, (nuc, bar) in enumerate(zip(sequence, bars)):
                ax.text(j, bar.get_height() + 0.02, nuc, ha='center', va='bottom', 
                       color=nuc_colors.get(nuc, 'gray'), fontweight='bold')
        
        # Добавляем название метода
        ax.set_title(f"{method}")
        
        # Добавляем подписи осей
        if i == n_methods - 1:
            ax.set_xlabel("Положение в последовательности")
        
        ax.set_ylabel("Значение атрибуции")
        
        # Устанавливаем пределы осей
        ax.set_xlim(0, len(normalized_attr))
        ax.set_ylim(0, 1.1)
    
    # Общее название
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Сохраняем или отображаем график
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def correlation_heatmap(attributions, methods, title, output_path=None):
    """
    Создает тепловую карту корреляций между различными методами атрибуции
    
    Args:
        attributions: словарь {метод: значения атрибуций}
        methods: список методов для отображения
        title: заголовок графика
        output_path: путь для сохранения изображения
    """
    # Вычисляем корреляции между методами
    corr_matrix = np.zeros((len(methods), len(methods)))
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            # Вычисляем коэффициент корреляции Спирмена
            corr, _ = spearmanr(attributions[method1], attributions[method2])
            corr_matrix[i, j] = corr
    
    # Создаем тепловую карту
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", 
                xticklabels=methods, yticklabels=methods, vmin=-1, vmax=1, center=0)
    plt.title(title)
    plt.tight_layout()
    
    # Сохраняем или отображаем
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def position_importance_plot(attributions, methods, title, output_path=None):
    """
    Создает график важности позиций в последовательности по данным атрибуций
    
    Args:
        attributions: словарь {метод: значения атрибуций}
        methods: список методов для отображения
        title: заголовок графика
        output_path: путь для сохранения изображения
    """
    plt.figure(figsize=(12, 6))
    
    # Нормализуем данные для сравнимости
    scaler = MinMaxScaler()
    
    for method in methods:
        # Нормализуем атрибуции
        normalized_attr = scaler.fit_transform(attributions[method].reshape(-1, 1)).reshape(-1)
        
        # Создаем линейный график
        plt.plot(range(len(normalized_attr)), normalized_attr, label=method, alpha=0.8)
    
    plt.xlabel("Положение в последовательности")
    plt.ylabel("Нормализованная важность")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Сохраняем или отображаем
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def summarize_attributions(attribution_data):
    """
    Создает сводку по атрибуциям различных методов
    
    Args:
        attribution_data: список словарей с данными атрибуций
        
    Returns:
        dict: сводные статистики
    """
    summary = {
        'total_sequences': len(attribution_data),
        'methods': {
            'saliency': {},
            'integrated_gradients': {},
            'deeplift': {}
        },
        'correlations': {
            'pearson': {},
            'spearman': {}
        }
    }
    
    # Методы для сравнения
    methods = ['saliency', 'integrated_gradients', 'deeplift']
    method_pairs = [
        ('saliency', 'integrated_gradients'),
        ('saliency', 'deeplift'),
        ('integrated_gradients', 'deeplift')
    ]
    
    # Вычисляем статистики по каждому методу
    for method in methods:
        values = []
        
        # Собираем значения атрибуций со всех последовательностей
        for data in attribution_data:
            values.extend(data[method].flatten())
        
        values = np.array(values)
        
        # Базовые статистики
        summary['methods'][method] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }
    
    # Вычисляем среднюю корреляцию между методами
    for pair in method_pairs:
        method1, method2 = pair
        
        pearson_corrs = []
        spearman_corrs = []
        
        for data in attribution_data:
            vals1 = data[method1].flatten()
            vals2 = data[method2].flatten()
            
            # Коэффициент корреляции Пирсона
            pearson, _ = pearsonr(vals1, vals2)
            pearson_corrs.append(pearson)
            
            # Коэффициент корреляции Спирмена
            spearman, _ = spearmanr(vals1, vals2)
            spearman_corrs.append(spearman)
        
        # Сохраняем средние значения
        summary['correlations']['pearson'][f"{method1}_vs_{method2}"] = float(np.mean(pearson_corrs))
        summary['correlations']['spearman'][f"{method1}_vs_{method2}"] = float(np.mean(spearman_corrs))
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Визуализация результатов интерпретации HyenaDNA с использованием Captum")
    parser.add_argument("--results_dir", type=str, required=True, help="Директория с результатами интерпретации")
    parser.add_argument("--output_dir", type=str, default="interpretation_comparison", help="Директория для сохранения визуализаций")
    
    args = parser.parse_args()
    
    # Создаем выходную директорию
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Загружаем данные
    try:
        attribution_data = load_attribution_data(args.results_dir)
        print(f"Загружены данные для {len(attribution_data)} последовательностей")
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return
    
    # Методы для сравнения
    methods = ['Saliency', 'Integrated Gradients', 'DeepLift']
    method_keys = ['saliency', 'integrated_gradients', 'deeplift']
    
    # Обрабатываем каждую последовательность
    for i, data in enumerate(attribution_data):
        sequence = data['sequence']
        sequence_name = f"sequence_{i+1}"
        
        # Преобразуем данные для визуализации
        attributions = {
            'Saliency': data['saliency'],
            'Integrated Gradients': data['integrated_gradients'],
            'DeepLift': data['deeplift']
        }
        
        print(f"Визуализация для {sequence_name}...")
        
        # Сравнительная визуализация атрибуций
        output_path = os.path.join(args.output_dir, f"{sequence_name}_comparison.png")
        visualize_sequence_with_attributions(
            sequence, attributions, methods,
            f"Сравнение методов атрибуции для {sequence_name}", output_path
        )
        
        # Тепловая карта корреляций
        output_path = os.path.join(args.output_dir, f"{sequence_name}_correlation.png")
        correlation_heatmap(
            attributions, methods,
            f"Корреляция между методами атрибуции для {sequence_name}", output_path
        )
        
        # График важности позиций
        output_path = os.path.join(args.output_dir, f"{sequence_name}_importance.png")
        position_importance_plot(
            attributions, methods,
            f"Важность позиций в последовательности {sequence_name}", output_path
        )
    
    # Создаем сводку
    print("Создание сводки по всем последовательностям...")
    summary = summarize_attributions(attribution_data)
    
    # Визуализируем средние корреляции
    plt.figure(figsize=(10, 6))
    
    method_pairs = [
        ('saliency_vs_integrated_gradients', 'Saliency vs IG'),
        ('saliency_vs_deeplift', 'Saliency vs DeepLift'),
        ('integrated_gradients_vs_deeplift', 'IG vs DeepLift')
    ]
    
    x = np.arange(len(method_pairs))
    width = 0.35
    
    # Строим столбцы для корреляций Пирсона и Спирмена
    pearson_values = [summary['correlations']['pearson'][pair[0]] for pair in method_pairs]
    spearman_values = [summary['correlations']['spearman'][pair[0]] for pair in method_pairs]
    
    plt.bar(x - width/2, pearson_values, width, label='Корреляция Пирсона', color='steelblue')
    plt.bar(x + width/2, spearman_values, width, label='Корреляция Спирмена', color='lightcoral')
    
    plt.xlabel('Пары методов')
    plt.ylabel('Средняя корреляция')
    plt.title('Средняя корреляция между методами атрибуции')
    plt.xticks(x, [pair[1] for pair in method_pairs])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(args.output_dir, "correlation_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Визуализации сохранены в {args.output_dir}")


if __name__ == "__main__":
    main() 