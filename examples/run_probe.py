#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np

# Добавляем корень проекта в Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.probe import ProbeTask, load_data_for_probing

def main():
    """
    Пример запуска пробинга на замороженных эмбеддингах ДНК моделей
    
    Варианты использования:
    1. Для извлечения эмбеддингов и обучения пробинг-модели:
       python run_probe.py --data_path путь/к/данным.csv --label_column метка --model_path путь/к/модели.ckpt
       
    2. Только для извлечения эмбеддингов:
       python run_probe.py --mode extract --data_path путь/к/данным.csv --label_column метка --model_path путь/к/модели.ckpt
       
    3. Только для пробинга на предварительно извлеченных эмбеддингах:
       python run_probe.py --mode probe --data_path путь/к/данным.csv --label_column метка --embeddings_path путь/к/эмбеддингам.npy
       
    4. Для использования модели HyenaDNA:
       python run_probe.py --data_path путь/к/данным.csv --label_column метка --model_path путь/к/весам.ckpt --config_path путь/к/config.json
    """
    parser = argparse.ArgumentParser(description='Пример запуска пробинга на замороженных эмбеддингах ДНК моделей')
    
    # Основные параметры
    parser.add_argument('--mode', type=str, choices=['extract', 'probe', 'full'], default='full',
                        help='Режим работы: extract - только извлечение эмбеддингов, '
                            'probe - только пробинг на предварительно извлеченных эмбеддингах, '
                            'full - полный процесс')
    parser.add_argument('--task_name', type=str, default='generic',
                        help='Название задачи пробинга (например, chromatin, expression, transcription)')
    
    # Параметры для данных
    parser.add_argument('--data_path', type=str, required=True,
                        help='Путь к файлу с данными (CSV, TSV, Excel)')
    parser.add_argument('--label_column', type=str, required=True,
                        help='Имя столбца с метками в данных')
    parser.add_argument('--seq_column', type=str, default='sequence',
                        help='Имя столбца с последовательностями ДНК в данных')
    
    # Параметры для модели и эмбеддингов
    parser.add_argument('--model_path', type=str,
                        help='Путь к весам модели (необходим для режимов extract и full)')
    parser.add_argument('--config_path', type=str,
                        help='Путь к конфигурационному файлу модели (для HyenaDNA)')
    parser.add_argument('--embeddings_path', type=str,
                        help='Путь к файлу с предварительно посчитанными эмбеддингами '
                            '(для сохранения в режиме extract или загрузки в режиме probe)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Устройство для вычислений (cuda или cpu)')
    
    # Параметры для пробинга
    parser.add_argument('--model_type', type=str, choices=['logistic', 'ridge'], default='logistic',
                        help='Тип линейной модели для пробинга (logistic для классификации, ridge для регрессии)')
    parser.add_argument('--cross_validation', action='store_true',
                        help='Использовать кросс-валидацию')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Количество фолдов для кросс-валидации')
    
    # Дополнительные параметры
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Размер пакета для извлечения эмбеддингов')
    parser.add_argument('--output_dir', type=str, default='probe_results',
                        help='Директория для сохранения результатов')
    parser.add_argument('--use_dummy_model', action='store_true',
                        help='Использовать фиктивную модель для тестирования')
    
    args = parser.parse_args()
    
    # Создание директории для результатов
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Инициализация задачи пробинга
    probe_task = ProbeTask(device=args.device)
    
    if args.mode in ['extract', 'full']:
        if not args.model_path and not args.use_dummy_model:
            print("Не указан путь к модели. Попытка использовать dummy_model.pt...")
            args.model_path = "weight/dummy_model.pt"
            args.use_dummy_model = True
        
        # Определяем тип модели
        is_hyenadna = False
        if args.config_path or (args.model_path and "hyenadna" in args.model_path):
            is_hyenadna = True
            
            # Если config_path не указан, но модель HyenaDNA, пытаемся найти config.json
            if not args.config_path and args.model_path:
                possible_config = os.path.join(os.path.dirname(args.model_path), "config.json")
                if os.path.exists(possible_config):
                    args.config_path = possible_config
                    print(f"Автоматически определен конфигурационный файл HyenaDNA: {args.config_path}")
            
        # Загрузка модели
        model_path = args.model_path
        print(f"Загрузка модели из {model_path}...")
        try:
            probe_task.load_model(model_path, args.config_path)
            print(f"Тип загруженной модели: {probe_task.model_type}")
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            if args.use_dummy_model:
                print("Попытка использовать dummy_model.pt для тестирования...")
                try:
                    probe_task.load_model("weight/dummy_model.pt")
                except Exception as e2:
                    print(f"Не удалось загрузить dummy модель: {e2}")
                    print("Продолжаем без модели, будут сгенерированы случайные эмбеддинги.")
        
        # Загрузка данных
        print(f"Загрузка данных из {args.data_path}...")
        sequences, labels = load_data_for_probing(
            args.data_path, args.label_column, args.seq_column
        )
        
        # Извлечение эмбеддингов
        print(f"Извлечение эмбеддингов из {len(sequences)} последовательностей...")
        try:
            embeddings = probe_task.extract_embeddings(sequences, batch_size=args.batch_size)
        except Exception as e:
            print(f"Ошибка при извлечении эмбеддингов: {e}")
            if args.use_dummy_model:
                print("Генерация случайных эмбеддингов для тестирования...")
                # Генерируем случайные эмбеддинги для тестирования
                embeddings = np.random.random((len(sequences), 256))  # Размерность 256 для примера
            else:
                raise
        
        # Сохранение эмбеддингов
        if args.embeddings_path:
            embeddings_path = args.embeddings_path
        else:
            embeddings_path = os.path.join(args.output_dir, 'embeddings.npy')
        
        print(f"Сохранение эмбеддингов в {embeddings_path}...")
        np.save(embeddings_path, embeddings)
        
        # Сохранение меток для последующего использования
        labels_path = os.path.join(args.output_dir, 'labels.npy')
        np.save(labels_path, labels)
    
    if args.mode in ['probe', 'full']:
        if args.mode == 'probe':
            # Загрузка предварительно извлеченных эмбеддингов
            if not args.embeddings_path:
                raise ValueError("Необходимо указать путь к эмбеддингам для режима probe")
            
            print(f"Загрузка предварительно извлеченных эмбеддингов из {args.embeddings_path}...")
            embeddings = probe_task.load_embeddings(args.embeddings_path)
            
            # Загрузка меток
            labels_path = os.path.join(os.path.dirname(args.embeddings_path), 'labels.npy')
            if os.path.exists(labels_path):
                print(f"Загрузка сохраненных меток из {labels_path}...")
                labels = np.load(labels_path)
            else:
                print(f"Метки не найдены. Загрузка из исходных данных {args.data_path}...")
                sequences, labels = load_data_for_probing(
                    args.data_path, args.label_column, args.seq_column
                )
        
        print(f"Проведение пробинга на эмбеддингах формы {embeddings.shape}...")
        
        # Выводим информацию о задаче и данных
        num_classes = len(np.unique(labels)) if args.model_type == 'logistic' else None
        if num_classes:
            print(f"Задача классификации с {num_classes} классами")
        else:
            print("Задача регрессии")
        
        # Обучение и оценка пробинг-модели
        if args.cross_validation:
            print(f"Запуск кросс-валидации с {args.cv_folds} фолдами...")
            results = probe_task.cross_validate(
                embeddings, labels, model_type=args.model_type, cv=args.cv_folds
            )
            
            print(f"Результаты кросс-валидации ({args.cv_folds} фолдов):")
            print(f"Средняя метрика: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
            
            # Сохранение результатов
            results_path = os.path.join(args.output_dir, f'{args.task_name}_cv_results.txt')
            with open(results_path, 'w') as f:
                f.write(f"Задача: {args.task_name}\n")
                f.write(f"Кросс-валидация ({args.cv_folds} фолдов):\n")
                f.write(f"Средняя метрика: {results['mean_score']:.4f} ± {results['std_score']:.4f}\n")
                f.write(f"Все значения: {', '.join([f'{x:.4f}' for x in results['all_scores']])}\n")
            
            # Визуализация результатов
            vis_path = os.path.join(args.output_dir, f'{args.task_name}_cv_results.png')
            metrics = {'fold_' + str(i+1): score for i, score in enumerate(results['all_scores'])}
            metrics['mean'] = results['mean_score']
            probe_task.visualize_results(metrics, output_path=vis_path, title=f'Пробинг: {args.task_name}')
        
        else:
            print("Запуск обучения на одном разделении данных...")
            model, metrics = probe_task.train_probe(
                embeddings, labels, model_type=args.model_type
            )
            
            print(f"Результаты обучения:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
            
            # Сохранение результатов
            results_path = os.path.join(args.output_dir, f'{args.task_name}_results.txt')
            with open(results_path, 'w') as f:
                f.write(f"Задача: {args.task_name}\n")
                f.write(f"Результаты обучения:\n")
                for metric_name, value in metrics.items():
                    f.write(f"{metric_name}: {value:.4f}\n")
            
            # Визуализация результатов
            vis_path = os.path.join(args.output_dir, f'{args.task_name}_results.png')
            probe_task.visualize_results(metrics, output_path=vis_path, title=f'Пробинг: {args.task_name}')
    
    print("Пример пробинга успешно завершен!")


if __name__ == "__main__":
    main() 