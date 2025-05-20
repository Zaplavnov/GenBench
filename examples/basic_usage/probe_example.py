#!/usr/bin/env python
"""
Пример использования пробинговых задач для анализа геномных моделей.
Этот пример демонстрирует, как использовать пробинговый анализ 
для исследования внутренних представлений моделей геномных последовательностей.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

# Добавляем корневую директорию в sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Импортируем необходимые модули из проекта GenBench
from src.models.sequence.hyena import HyenaModel
from src.utils.tokenizer_dna import DNATokenizer
from src.probe import LinearProbe

def generate_sample_data(num_samples=100, seq_length=50, embedding_dim=256):
    """
    Генерация синтетических данных для примера пробинговой задачи.
    
    Args:
        num_samples: количество образцов
        seq_length: длина последовательности
        embedding_dim: размерность эмбеддинга
        
    Returns:
        embeddings: эмбеддинги (имитация выхода модели)
        labels: метки для задачи классификации
    """
    # Генерируем синтетические эмбеддинги
    embeddings = np.random.randn(num_samples, embedding_dim)
    
    # Генерируем синтетические метки (бинарные классы)
    # В этом примере мы связываем класс с наличием определенного паттерна в эмбеддингах
    pattern = np.random.randn(embedding_dim)
    similarity = np.dot(embeddings, pattern)
    labels = (similarity > 0).astype(int)
    
    return embeddings, labels

def main():
    print("Пример использования пробинговых задач для анализа геномных моделей")
    
    # 1. Сначала мы должны получить эмбеддинги от модели (для простоты используем синтетические данные)
    print("Генерация синтетических данных для примера...")
    embeddings, labels = generate_sample_data(num_samples=100, embedding_dim=64)
    
    # 2. Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )
    
    print(f"Размеры данных - Обучение: {X_train.shape}, Тест: {X_test.shape}")
    
    # 3. Обучаем линейный пробинговый классификатор
    print("Обучение линейного пробингового классификатора...")
    probe = LinearProbe(
        input_dim=X_train.shape[1],
        task_type="classification",
        num_classes=2
    )
    
    # Обучаем пробинговую модель
    metrics = probe.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        max_epochs=10
    )
    
    # 4. Оцениваем пробинговую модель на тестовых данных
    accuracy = probe.evaluate(X_test, y_test)
    print(f"Точность пробинговой модели на тестовых данных: {accuracy:.4f}")
    
    # 5. Визуализируем результаты обучения
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(metrics['train_loss']) + 1), metrics['train_loss'], label='Train Loss')
    if 'val_loss' in metrics:
        plt.plot(range(1, len(metrics['val_loss']) + 1), metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Probe Training Progress')
    plt.legend()
    
    # Сохраняем график
    output_dir = "probe_results/example"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "probe_training.png"))
    print(f"График сохранен в {os.path.join(output_dir, 'probe_training.png')}")
    
    # 6. Анализируем важность признаков
    if hasattr(probe.model, 'coef_'):
        weights = probe.model.coef_
        plt.figure(figsize=(12, 6))
        plt.bar(range(weights.shape[1]), weights[0])
        plt.xlabel('Feature Index')
        plt.ylabel('Weight')
        plt.title('Feature Importance in Probe Model')
        plt.savefig(os.path.join(output_dir, "feature_importance.png"))
        print(f"График важности признаков сохранен в {os.path.join(output_dir, 'feature_importance.png')}")
    
if __name__ == "__main__":
    main() 