#!/usr/bin/env python
"""
Пример использования инструментов интерпретируемости с библиотекой Captum.
Этот пример демонстрирует, как использовать Captum для интерпретации 
предсказаний моделей на ДНК-последовательностях.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import captum
    from captum.attr import IntegratedGradients, DeepLift, Saliency
except ImportError:
    print("Для работы примера требуется установить библиотеку Captum:")
    print("pip install captum")
    sys.exit(1)

# Добавляем корневую директорию в sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Импортируем необходимые модули из проекта GenBench
from src.models.sequence.hyena import HyenaModel
from src.utils.tokenizer_dna import DNATokenizer
from src.utils.interpret import plot_dna_attributions

# Создаем простую модель-обертку для интерпретации
class DNAInterpreter(torch.nn.Module):
    """
    Обертка для модели, которая позволяет получать атрибуции для входных данных.
    """
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        
    def forward(self, input_ids):
        """
        Прямой проход через модель.
        
        Args:
            input_ids: тензор токенов ДНК-последовательности
            
        Returns:
            Предсказанные классы/логиты
        """
        embeddings = self.model(input_ids)
        # Для примера используем среднее значение эмбеддингов по всем позициям
        # и затем применяем линейный слой для получения логитов
        pooled = embeddings.mean(dim=1)  # [batch, seq_len, dim] -> [batch, dim]
        
        # Простой классификатор (в реальности нужно обучить)
        logits = torch.nn.functional.linear(pooled, torch.randn(2, pooled.shape[-1]).to(pooled.device))
        return logits
    
    def predict(self, sequence):
        """
        Получает предсказание для ДНК-последовательности.
        
        Args:
            sequence: строка с ДНК-последовательностью
            
        Returns:
            Предсказанные вероятности классов
        """
        tokens = self.tokenizer.tokenize(sequence)
        token_ids = torch.tensor(tokens).unsqueeze(0).to(next(self.model.parameters()).device)
        with torch.no_grad():
            logits = self(token_ids)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs
    
    def get_attributions(self, sequence, method="ig", target_class=0):
        """
        Получает атрибуции для ДНК-последовательности.
        
        Args:
            sequence: строка с ДНК-последовательностью
            method: метод атрибуции ('ig', 'dl', или 'saliency')
            target_class: целевой класс для атрибуции
            
        Returns:
            Атрибуции для каждого нуклеотида
        """
        tokens = self.tokenizer.tokenize(sequence)
        token_ids = torch.tensor(tokens).unsqueeze(0).to(next(self.model.parameters()).device)
        token_ids.requires_grad_()
        
        # Выбираем метод атрибуции
        if method == "ig":
            attr_func = IntegratedGradients(self)
        elif method == "dl":
            attr_func = DeepLift(self)
        elif method == "saliency":
            attr_func = Saliency(self)
        else:
            raise ValueError(f"Неизвестный метод атрибуции: {method}")
        
        # Получаем атрибуции
        if method == "saliency":
            attributions = attr_func.attribute(token_ids, target=target_class)
        else:
            baseline = torch.zeros_like(token_ids)
            attributions = attr_func.attribute(token_ids, baseline, target=target_class)
        
        return attributions.squeeze(0).detach().cpu().numpy()

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
    
    try:
        # Загружаем модель
        print("Загрузка модели HyenaDNA...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = HyenaModel.from_pretrained(
            checkpoint_path=model_path,
            config_path=config_path
        )
        model = model.to(device)
        model.eval()
        
        # Создаем токенизатор
        tokenizer = DNATokenizer()
        
        # Создаем интерпретатор
        interpreter = DNAInterpreter(model, tokenizer)
        interpreter.eval()
        
        # Пример ДНК-последовательности (промотор гена)
        dna_sequence = "ACTGCTAGCTACGTAGCTACGTACGTACGTACGTACGTACGTACGTACGTACGTAGCTACGTACGTACGTACGTACGT"
        print(f"Анализируем ДНК-последовательность: {dna_sequence[:20]}...")
        
        # Получаем предсказание
        probs = interpreter.predict(dna_sequence)
        pred_class = probs.argmax(dim=-1).item()
        pred_prob = probs[0, pred_class].item()
        print(f"Предсказанный класс: {pred_class}, Вероятность: {pred_prob:.4f}")
        
        # Получаем атрибуции для каждого метода
        methods = ["ig", "dl", "saliency"]
        attributions = {}
        
        for method in methods:
            print(f"Вычисление атрибуций с методом {method}...")
            attr = interpreter.get_attributions(dna_sequence, method=method, target_class=pred_class)
            attributions[method] = attr
            
        # Визуализируем атрибуции
        output_dir = "interpret_results_example"
        os.makedirs(output_dir, exist_ok=True)
        
        # Визуализируем атрибуции для каждого метода
        for method, attr in attributions.items():
            plt.figure(figsize=(15, 5))
            plt.bar(range(len(attr)), attr)
            plt.title(f"Атрибуции метода {method.upper()} для ДНК-последовательности")
            plt.xlabel("Позиция")
            plt.ylabel("Значение атрибуции")
            plt.savefig(os.path.join(output_dir, f"attribution_{method}.png"))
            
        # Визуализируем сравнение методов
        plt.figure(figsize=(15, 10))
        for i, (method, attr) in enumerate(attributions.items()):
            plt.subplot(len(methods), 1, i+1)
            plt.bar(range(len(attr)), attr)
            plt.title(f"Метод {method.upper()}")
            plt.ylabel("Значение атрибуции")
            if i == len(methods) - 1:
                plt.xlabel("Позиция")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "attribution_comparison.png"))
        
        print(f"Результаты сохранены в директории {output_dir}")
        
    except Exception as e:
        print(f"Ошибка при выполнении примера: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 