#!/usr/bin/env python

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from captum.attr import Saliency, IntegratedGradients, DeepLift
import argparse
from sklearn.preprocessing import MinMaxScaler

# Добавляем путь к корневой директории
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Импортируем модули проекта
try:
    from hyena_dna.standalone import load_hyenadna
    from src.utils.tokenizer_dna import one_hot_encode_dna
    HYENA_MODULES_LOADED = True
except ImportError:
    print("Не удалось загрузить модули hyena_dna, будет использоваться только HuggingFace версия")
    HYENA_MODULES_LOADED = False


def load_huggingface_model(model_path, device="cuda"):
    """
    Загружает модель HyenaDNA в формате HuggingFace
    
    Args:
        model_path: путь к директории с моделью HuggingFace
        device: устройство для инференса
        
    Returns:
        model: загруженная модель
        tokenizer: токенизатор
    """
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        raise ImportError("Для использования моделей HuggingFace требуется установить transformers")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    device_map = "auto" if device == "cuda" and torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        torch_dtype=torch.float32, 
        device_map=device_map, 
        trust_remote_code=True
    )
    
    return model, tokenizer


class HuggingFaceForwardWrapper(torch.nn.Module):
    """Обертка для модели HuggingFace для совместимости с Captum"""
    
    def __init__(self, model, tokenizer, target_class=0):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.target_class = target_class
    
    def forward(self, input_tensor):
        """
        Форматирует вход для HuggingFace модели
        
        Args:
            input_tensor: тензор one-hot кодированных последовательностей [batch_size, seq_len, channels]
            
        Returns:
            output: логиты или вероятности [batch_size, num_classes]
        """
        # Преобразуем one-hot обратно в последовательность
        if input_tensor.dim() == 3:
            batch_size, seq_len, n_channels = input_tensor.shape
            
            # Для упрощения берем индекс максимального значения в каждой позиции
            indices = torch.argmax(input_tensor, dim=2).cpu().numpy()
            
            # Конвертируем индексы в символы (0=A, 1=C, 2=G, 3=T, 4=N)
            mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}
            sequences = []
            
            for seq_indices in indices:
                sequence = ''.join([mapping[idx] for idx in seq_indices])
                sequences.append(sequence)
            
            # Токенизируем текст с помощью HuggingFace токенизатора
            inputs = self.tokenizer(sequences, return_tensors="pt", padding=True).to(input_tensor.device)
            
            # Выполняем прямой проход через модель
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Получаем логиты
            logits = outputs.logits
            
            # Добавляем softmax для получения вероятностей
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            return probs
        else:
            raise ValueError(f"Ожидается вход с размерностью 3, получено: {input_tensor.dim()}")


class HyenaDNAForwardWrapper(torch.nn.Module):
    """Wrapper для модели HyenaDNA для совместимости с Captum"""
    
    def __init__(self, model, target_class=0):
        super().__init__()
        self.model = model
        self.target_class = target_class
    
    def forward(self, input_tensor):
        """
        Форматирует вход для HyenaDNA и возвращает вероятности или предсказания
        
        Args:
            input_tensor: тензор с формой [batch_size, seq_len, channels]
            
        Returns:
            тензор с формой [batch_size, num_classes]
        """
        # Преобразуем one-hot обратно в последовательность
        if input_tensor.dim() == 3:
            batch_size, seq_len, n_channels = input_tensor.shape
            
            # Для упрощения берем индекс максимального значения в каждой позиции
            indices = torch.argmax(input_tensor, dim=2).cpu().numpy()
            
            # Конвертируем индексы в символы (0=A, 1=C, 2=G, 3=T, 4=N)
            mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}
            sequences = []
            
            for seq_indices in indices:
                sequence = ''.join([mapping[idx] for idx in seq_indices])
                sequences.append(sequence)
            
            # Форматируем вход для HyenaDNA
            inputs = {"sequence": sequences}
            
            # Выполняем прямой проход через модель
            with torch.no_grad():
                outputs = self.model.model(inputs)
            
            # Если модель возвращает скрытые состояния, берем только логиты
            if isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
            else:
                logits = outputs
            
            # Добавляем softmax, если модель возвращает логиты
            if not torch.allclose(torch.sum(torch.nn.functional.softmax(logits, dim=-1), dim=-1), 
                                torch.ones(logits.shape[0], device=logits.device)):
                probs = torch.nn.functional.softmax(logits, dim=-1)
            else:
                probs = logits
            
            return probs
        else:
            raise ValueError(f"Ожидается вход с размерностью 3, получено: {input_tensor.dim()}")


def compute_captum_attribution(model_wrapper, input_tensor, method="saliency", target=0, n_steps=50, baseline_type="zero"):
    """
    Вычисляет атрибуции с использованием методов Captum
    
    Args:
        model_wrapper: обертка модели, совместимая с Captum
        input_tensor: входной тензор для анализа
        method: метод атрибуции ('saliency', 'ig', 'dl')
        target: целевой класс для атрибуции
        n_steps: количество шагов для Integrated Gradients
        baseline_type: тип базовой линии ('zero', 'random')
        
    Returns:
        numpy.ndarray: значения атрибуции
    """
    model_wrapper.eval()
    input_tensor.requires_grad = True
    
    # Создаем базовую линию
    if baseline_type == "zero":
        baseline = torch.zeros_like(input_tensor)
    elif baseline_type == "random":
        baseline = torch.rand_like(input_tensor) * 0.1
    else:
        raise ValueError(f"Неизвестный тип базовой линии: {baseline_type}")
    
    # Выбираем метод атрибуции
    if method.lower() == "saliency":
        explainer = Saliency(model_wrapper)
        attr = explainer.attribute(input_tensor, target=target)
    elif method.lower() == "ig":
        explainer = IntegratedGradients(model_wrapper)
        attr = explainer.attribute(input_tensor, baselines=baseline, target=target, n_steps=n_steps)
    elif method.lower() == "deeplift" or method.lower() == "dl":
        explainer = DeepLift(model_wrapper)
        attr = explainer.attribute(input_tensor, baselines=baseline, target=target)
    else:
        raise ValueError(f"Неизвестный метод атрибуции: {method}")
    
    # Суммируем атрибуции по каналам
    attr_sum = attr.sum(dim=2).detach().cpu().numpy()
    
    return attr_sum


def visualize_sequence_attribution(sequence, attribution, method, output_path=None, title=None):
    """
    Визуализирует атрибуции для последовательности ДНК
    
    Args:
        sequence: строка последовательности ДНК
        attribution: значения атрибуции (numpy array)
        method: метод атрибуции (для заголовка)
        output_path: путь для сохранения изображения
        title: заголовок графика
    """
    # Нормализуем атрибуции для лучшей визуализации
    scaler = MinMaxScaler()
    normalized_attr = scaler.fit_transform(attribution.reshape(-1, 1)).reshape(-1)
    
    # Создаем фигуру
    plt.figure(figsize=(15, 5))
    
    # Создаем цветовую карту для нуклеотидов
    nuc_colors = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red', 'N': 'gray'}
    
    # Отображаем атрибуции
    bars = plt.bar(range(len(normalized_attr)), normalized_attr, width=0.8, alpha=0.6)
    
    # Добавляем символы нуклеотидов с цветовой кодировкой
    if len(sequence) == len(normalized_attr):
        for i, (nuc, bar) in enumerate(zip(sequence, bars)):
            color = nuc_colors.get(nuc, 'gray')
            plt.text(i, bar.get_height() + 0.02, nuc, ha='center', va='bottom', color=color, fontweight='bold')
    
    # Устанавливаем заголовок и подписи осей
    if title:
        plt.title(title)
    else:
        plt.title(f"Атрибуция последовательности ДНК - {method}")
    
    plt.xlabel("Положение в последовательности")
    plt.ylabel("Значение атрибуции (нормализованное)")
    plt.tight_layout()
    
    # Сохраняем или показываем изображение
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def compare_attribution_methods(sequence, model, model_type="hyenadna", tokenizer=None, target_class=0, output_dir="interpret_results_hyenadna"):
    """
    Сравнивает различные методы атрибуции на одной последовательности
    
    Args:
        sequence: строка последовательности ДНК
        model: модель HyenaDNA или HuggingFace
        model_type: тип модели ("hyenadna" или "huggingface")
        tokenizer: токенизатор для huggingface модели
        target_class: целевой класс для атрибуции
        output_dir: директория для сохранения результатов
    """
    # Создаем директорию для результатов
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем обертку для модели
    if model_type == "hyenadna":
        model_wrapper = HyenaDNAForwardWrapper(model, target_class=target_class)
    elif model_type == "huggingface":
        if tokenizer is None:
            raise ValueError("Для модели HuggingFace необходим токенизатор")
        model_wrapper = HuggingFaceForwardWrapper(model, tokenizer, target_class=target_class)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    model_wrapper.eval()
    
    # Преобразуем последовательность в one-hot
    one_hot = one_hot_encode_dna(sequence, include_n=True)
    one_hot = torch.from_numpy(one_hot).float().unsqueeze(0)  # Добавляем размерность пакета
    
    if torch.cuda.is_available():
        one_hot = one_hot.cuda()
    
    # Методы атрибуции для сравнения
    methods = {
        "Saliency": "saliency",
        "Integrated Gradients": "ig",
        "DeepLift": "dl"
    }
    
    # Вычисляем атрибуции для каждого метода
    all_attributions = {}
    for name, method in methods.items():
        print(f"Вычисление атрибуций методом {name}...")
        attr = compute_captum_attribution(model_wrapper, one_hot, method=method, target=target_class)
        all_attributions[name] = attr[0]  # Берем только первый элемент пакета
        
        # Визуализируем и сохраняем отдельные графики
        output_path = os.path.join(output_dir, f"attribution_{method.lower()}.png")
        visualize_sequence_attribution(sequence, attr[0], name, output_path)
    
    # Создаем сравнительный график всех методов
    plt.figure(figsize=(15, 10))
    
    for i, (name, attr) in enumerate(all_attributions.items(), 1):
        plt.subplot(len(methods), 1, i)
        
        # Нормализуем атрибуции для сравнимости
        scaler = MinMaxScaler()
        norm_attr = scaler.fit_transform(attr.reshape(-1, 1)).reshape(-1)
        
        plt.bar(range(len(norm_attr)), norm_attr, alpha=0.7)
        plt.title(f"Метод {name}")
        
        if i == len(methods):
            plt.xlabel("Положение в последовательности")
        
        plt.ylabel("Атрибуция")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Сохраняем числовые данные
    np.savez(
        os.path.join(output_dir, "attribution_data.npz"),
        sequence=sequence,
        saliency=all_attributions["Saliency"],
        integrated_gradients=all_attributions["Integrated Gradients"],
        deeplift=all_attributions["DeepLift"]
    )
    
    return all_attributions


def main():
    parser = argparse.ArgumentParser(description="Интерпретация модели HyenaDNA с использованием методов Captum")
    parser.add_argument("--weights_path", type=str, help="Путь к весам модели HyenaDNA (для оригинального формата)")
    parser.add_argument("--config_path", type=str, help="Путь к конфигурации модели (для оригинального формата)")
    parser.add_argument("--model_path", type=str, help="Путь к модели HuggingFace HyenaDNA")
    parser.add_argument("--hf_format", type=bool, default=False, help="Использовать формат HuggingFace")
    parser.add_argument("--sequence", type=str, help="Последовательность ДНК для анализа")
    parser.add_argument("--fasta", type=str, help="Путь к FASTA файлу с последовательностями")
    parser.add_argument("--output_dir", type=str, default="interpret_results_hyenadna", help="Директория для сохранения результатов")
    parser.add_argument("--target_class", type=int, default=0, help="Целевой класс для атрибуции")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Устройство для вычислений")
    
    args = parser.parse_args()
    
    # Загружаем модель
    if args.hf_format or args.model_path:
        if not args.model_path:
            parser.error("Для HuggingFace формата требуется указать --model_path")
        print(f"Загрузка модели HuggingFace из {args.model_path}...")
        model, tokenizer = load_huggingface_model(args.model_path, device=args.device)
        model_type = "huggingface"
    else:
        if not HYENA_MODULES_LOADED:
            parser.error("Не удалось загрузить модули hyena_dna для работы с оригинальной моделью")
        if not args.weights_path:
            parser.error("Для оригинального формата требуется указать --weights_path")
        print(f"Загрузка модели HyenaDNA из {args.weights_path}...")
        model = load_hyenadna(args.weights_path, args.config_path, device=args.device)
        tokenizer = None
        model_type = "hyenadna"
    
    # Получаем последовательности для анализа
    sequences = []
    
    if args.sequence:
        sequences.append(args.sequence)
    
    if args.fasta:
        try:
            from Bio import SeqIO
            print(f"Загрузка последовательностей из {args.fasta}...")
            for record in SeqIO.parse(args.fasta, "fasta"):
                sequences.append(str(record.seq))
        except ImportError:
            print("Для чтения FASTA файлов требуется библиотека Biopython")
            if not args.sequence:
                print("Ни одной последовательности не предоставлено. Выход.")
                return
    
    if not sequences:
        # Используем тестовую последовательность, если ничего не предоставлено
        test_seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
        sequences = [test_seq]
        print("Используем тестовую последовательность:", sequences[0])
    
    # Анализируем каждую последовательность
    for i, seq in enumerate(sequences):
        print(f"Анализ последовательности {i+1}/{len(sequences)}...")
        
        # Создаем поддиректорию для этой последовательности
        seq_output_dir = os.path.join(args.output_dir, f"sequence_{i+1}")
        os.makedirs(seq_output_dir, exist_ok=True)
        
        # Анализируем последовательность
        compare_attribution_methods(
            sequence=seq,
            model=model,
            model_type=model_type,
            tokenizer=tokenizer,
            target_class=args.target_class,
            output_dir=seq_output_dir
        )
    
    print(f"Анализ завершен. Результаты сохранены в {args.output_dir}")


if __name__ == "__main__":
    main() 