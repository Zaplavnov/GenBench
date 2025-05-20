#!/usr/bin/env python
"""
Пример использования модели HyenaDNA для обработки ДНК-последовательности.
Этот пример демонстрирует, как загрузить предобученную модель HyenaDNA,
обработать ДНК-последовательность и получить предсказания.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Добавляем корневую директорию в sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Импортируем необходимые модули из проекта GenBench
from src.models.sequence.hyena import HyenaModel
from src.utils.tokenizer_dna import DNATokenizer

def main():
    # Параметры
    model_path = "models/hyenadna-tiny-1k-seqlen/weights.ckpt"  # Путь к весам модели
    config_path = "models/hyenadna-tiny-1k-seqlen/config.json"  # Путь к конфигурации модели
    
    # Проверяем наличие весов модели
    if not os.path.exists(model_path):
        print(f"Модель не найдена по пути {model_path}")
        print("Сначала загрузите модель с помощью скрипта download_models.py:")
        print("python download_models.py --model_type tiny-1k-seqlen")
        return
    
    # Загружаем модель
    print("Загрузка модели HyenaDNA...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Загружаем модель
        model = HyenaModel.from_pretrained(
            checkpoint_path=model_path,
            config_path=config_path
        )
        model = model.to(device)
        model.eval()
        print(f"Модель успешно загружена на устройство: {device}")
        
        # Создаем токенизатор
        tokenizer = DNATokenizer()
        
        # Пример ДНК-последовательности (промотор гена TP53)
        dna_sequence = "ACTGCTAGCTACGTAGCTACGTACGTACGTACGTACGTACGTACGTACGTACGTAGCTACGTACGTACGTACGTACGT"
        print(f"Обрабатываем ДНК-последовательность: {dna_sequence[:20]}...")
        
        # Токенизируем последовательность
        tokens = tokenizer.tokenize(dna_sequence)
        token_ids = torch.tensor(tokens).unsqueeze(0).to(device)  # Добавляем размерность батча
        
        # Получаем эмбеддинги
        with torch.no_grad():
            embeddings = model(token_ids)
        
        print(f"Получены эмбеддинги формы: {embeddings.shape}")
        
        # Пример использования эмбеддингов: визуализируем первые 5 значений для первого нуклеотида
        print("Первые 5 значений эмбеддинга для первого нуклеотида:")
        print(embeddings[0, 0, :5].cpu().numpy())
        
    except Exception as e:
        print(f"Ошибка при загрузке или выполнении модели: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 