import os
import sys
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from captum.attr import Saliency, IntegratedGradients, DeepLift
from sklearn.preprocessing import MinMaxScaler
from Bio import SeqIO

# Токенизация ДНК последовательностей
def one_hot_encode_dna(sequence, include_n=True):
    """
    One-hot encode a DNA sequence.

    Args:
        sequence: DNA sequence string
        include_n: Whether to include N as a separate channel (default: True)
                   If False, N will be encoded as [0.25, 0.25, 0.25, 0.25]

    Returns:
        numpy array with one-hot encoded sequence
    """
    sequence = sequence.upper()
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4 if include_n else None}
    num_channels = 5 if include_n else 4
    
    # Создаем массив
    seq_length = len(sequence)
    one_hot = np.zeros((seq_length, num_channels))
    
    # Заполняем массив
    for i, nuc in enumerate(sequence):
        if nuc in mapping and mapping[nuc] is not None:
            one_hot[i, mapping[nuc]] = 1.0
        elif not include_n and nuc == 'N':
            # If N is not included as a separate channel, distribute evenly
            one_hot[i, :] = 0.25
    
    return one_hot


class SimpleModelWrapper(torch.nn.Module):
    """
    Простая обертка для модели pytorch для использования с Captum
    """
    def __init__(self, model, target_class=0):
        super().__init__()
        self.model = model
        self.target_class = target_class
    
    def forward(self, input_tensor):
        """
        Forward pass для Captum
        """
        # Получаем логиты
        logits = self.model(input_tensor)
        
        # Применяем softmax
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        return probs


class HyenaDNAModel(torch.nn.Module):
    """
    Простая модель для тестирования интерпретируемости
    с архитектурой, похожей на HyenaDNA
    """
    def __init__(self, d_model=128, num_classes=2):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Создаем простую сеть
        self.embedding = torch.nn.Linear(5, d_model)  # 5 каналов (A, C, G, T, N)
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            torch.nn.ReLU(),
        )
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.classifier = torch.nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: One-hot encoded DNA sequence [batch_size, seq_length, channels]
            
        Returns:
            logits: [batch_size, num_classes]
        """
        batch_size, seq_length, channels = x.shape
        
        # Embedding
        x = self.embedding(x)  # [batch_size, seq_length, d_model]
        
        # Перестановка размерностей для Conv1d
        x = x.transpose(1, 2)  # [batch_size, d_model, seq_length]
        
        # Encoder
        x = self.encoder(x)  # [batch_size, d_model, seq_length]
        
        # Pooling
        x = self.pool(x).squeeze(2)  # [batch_size, d_model]
        
        # Classifier
        logits = self.classifier(x)  # [batch_size, num_classes]
        
        return logits


def load_config(config_path):
    """
    Загружает конфигурацию модели
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def create_model(config, device="cuda"):
    """
    Создает модель на основе конфигурации
    """
    d_model = config.get("d_model", 128)
    num_classes = 2  # Для простоты
    
    model = HyenaDNAModel(d_model=d_model, num_classes=num_classes)
    model = model.to(device)
    model.eval()
    
    return model


def compute_captum_attribution(model_wrapper, input_tensor, method="saliency", target=0, n_steps=50):
    """
    Вычисляет атрибуции с использованием методов Captum
    
    Args:
        model_wrapper: обертка модели, совместимая с Captum
        input_tensor: входной тензор для анализа
        method: метод атрибуции ('saliency', 'ig', 'dl')
        target: целевой класс для атрибуции
        n_steps: количество шагов для Integrated Gradients
        
    Returns:
        numpy.ndarray: значения атрибуции
    """
    model_wrapper.eval()
    input_tensor.requires_grad = True
    
    # Создаем базовую линию
    baseline = torch.zeros_like(input_tensor)
    
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
    
    # Добавляем символы нуклеотидов с цветовой кодировкой, если последовательность не слишком длинная
    if len(sequence) <= 100:
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


def compare_attribution_methods(sequence, model, target_class=0, output_dir="interpret_results_hyenadna", device="cuda"):
    """
    Сравнивает различные методы атрибуции на одной последовательности
    
    Args:
        sequence: строка последовательности ДНК
        model: модель
        target_class: целевой класс для атрибуции
        output_dir: директория для сохранения результатов
        device: устройство для вычислений
    """
    # Создаем директорию для результатов
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем обертку для модели
    model_wrapper = SimpleModelWrapper(model, target_class=target_class)
    model_wrapper.eval()
    
    # Преобразуем последовательность в one-hot
    one_hot = one_hot_encode_dna(sequence, include_n=True)
    one_hot = torch.from_numpy(one_hot).float().unsqueeze(0)  # Добавляем размерность пакета
    
    if device == "cuda" and torch.cuda.is_available():
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


def main(fasta_path="real_dna_sequences.fa", config_path=None, output_dir="interpret_results_hyenadna", device="cuda"):
    """
    Основная функция для запуска интерпретации
    """
    # Загружаем конфигурацию модели
    if config_path is not None:
        config = load_config(config_path)
    else:
        # Дефолтная конфигурация
        config = {"d_model": 128, "num_classes": 2}
    
    # Создаем модель
    model = create_model(config, device=device)
    print(f"Создана тестовая модель с {sum(p.numel() for p in model.parameters())} параметрами")
    
    # Загружаем последовательности из FASTA файла
    sequences = []
    names = []
    
    if os.path.exists(fasta_path):
        print(f"Загрузка последовательностей из {fasta_path}...")
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences.append(str(record.seq))
            names.append(record.id)
    else:
        print(f"Файл {fasta_path} не найден. Используем тестовую последовательность.")
        sequences = ["ACGTACGTACGTACGTACGT" * 5]
        names = ["test_sequence"]
    
    # Анализируем каждую последовательность
    for i, (seq, name) in enumerate(zip(sequences, names)):
        print(f"Анализ последовательности {i+1}/{len(sequences)}: {name}...")
        
        # Создаем поддиректорию для этой последовательности
        seq_output_dir = os.path.join(output_dir, name)
        os.makedirs(seq_output_dir, exist_ok=True)
        
        # Анализируем последовательность
        compare_attribution_methods(
            sequence=seq,
            model=model,
            target_class=0,
            output_dir=seq_output_dir,
            device=device
        )
    
    print(f"Анализ завершен. Результаты сохранены в {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Интерпретация моделей ДНК с использованием методов Captum")
    parser.add_argument("--fasta", type=str, default="real_dna_sequences.fa", help="Путь к FASTA файлу с последовательностями")
    parser.add_argument("--config_path", type=str, help="Путь к конфигурации модели")
    parser.add_argument("--output_dir", type=str, default="interpret_results_hyenadna", help="Директория для сохранения результатов")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Устройство для вычислений")
    
    args = parser.parse_args()
    
    main(
        fasta_path=args.fasta,
        config_path=args.config_path,
        output_dir=args.output_dir,
        device=args.device
    ) 