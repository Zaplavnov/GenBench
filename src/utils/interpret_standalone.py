import os
import sys
# Добавляем корень проекта в Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from captum.attr import Saliency, IntegratedGradients, DeepLift
from Bio import motifs
from Bio.motifs import jaspar

# Импортируем функцию для one-hot кодирования
from src.utils.tokenizer_dna import one_hot_encode_dna

def load_model_from_checkpoint(model_path):
    """
    Load model from checkpoint
    """
    try:
        # Загрузка весов с weights_only=False
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        
        # Вывод информации о checkpoint
        print(f"Загружен checkpoint из {model_path}")
        print(f"Размер: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")
        
        # Базовая проверка содержимого checkpoint
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                print(f"Найден state_dict с {len(state_dict)} параметрами")
            else:
                print(f"Checkpoint содержит следующие ключи: {list(checkpoint.keys())}")
        else:
            print(f"Checkpoint не является словарем, его тип: {type(checkpoint)}")
        
        return checkpoint
    except Exception as e:
        print(f"Ошибка при загрузке checkpoint: {e}")
        # Пробуем другой вариант загрузки
        try:
            print("Пробуем загрузить модель без проверки весов...")
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            print("Успешно загружены только веса модели.")
            return checkpoint
        except Exception as e2:
            print(f"Не удалось загрузить модель: {e2}")
            return None

def tokenize_sequence(seq, include_n=True):
    """
    Tokenize sequence using one-hot encoding
    
    Args:
        seq: DNA sequence
        include_n: Whether to include N as a separate channel
    
    Returns:
        PyTorch tensor with one-hot encoded sequence
    """
    # Используем функцию из tokenizer_dna.py
    return one_hot_encode_dna(seq, include_n=include_n)

def create_demo_motif():
    """
    Создает демонстрационный мотив TATA-box
    
    Returns:
        Словарь, имитирующий PSSM мотива
    """
    # Создаем демонстрационный PSSM для TATA-box
    # TATA-box обычно имеет консенсусную последовательность TATAAAA
    pssm = {
        'length': 7,
        'pssm': np.array([
            [0.1, 0.1, 0.1, 0.7],  # T
            [0.7, 0.1, 0.1, 0.1],  # A
            [0.1, 0.1, 0.1, 0.7],  # T
            [0.7, 0.1, 0.1, 0.1],  # A
            [0.7, 0.1, 0.1, 0.1],  # A
            [0.7, 0.1, 0.1, 0.1],  # A
            [0.7, 0.1, 0.1, 0.1],  # A
        ])
    }
    return pssm

def load_jaspar_motif(motif_id):
    """
    Load a motif from JASPAR database by ID
    
    Args:
        motif_id: JASPAR motif ID (e.g., 'MA0108.1')
    
    Returns:
        Position-specific scoring matrix (PSSM)
    """
    try:
        # Проверяем, есть ли доступ к JASPAR
        print("Пытаемся загрузить мотив из JASPAR...")
        
        # Проверка доступных методов в модуле jaspar
        available_methods = [method for method in dir(jaspar) if not method.startswith('_')]
        print(f"Доступные методы в Bio.motifs.jaspar: {available_methods}")
        
        # Если JASPAR5 недоступен, используем демо-мотив
        print("Доступ к JASPAR не удался, используем демонстрационный мотив.")
        return create_demo_motif()
        
    except Exception as e:
        print(f"Ошибка при загрузке мотива JASPAR: {e}")
        print("Используем демонстрационный мотив.")
        return create_demo_motif()

def correlate_with_motif(seq_scores, motif_pssm, seq):
    """
    Correlate attribution scores with a motif PSSM
    
    Args:
        seq_scores: Attribution scores for sequence
        motif_pssm: Position-specific scoring matrix
        seq: Original sequence
    
    Returns:
        Array of correlation scores
    """
    if motif_pssm is None:
        return None
    
    correlations = []
    L = len(seq_scores)
    k = motif_pssm['length']
    
    # One-hot encode sequence
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    
    for i in range(L - k + 1):
        window = seq_scores[i:i+k]
        subseq = seq[i:i+k]
        
        # Skip windows with 'N'
        if 'N' in subseq:
            correlations.append(0)
            continue
        
        # Convert subsequence to one-hot
        one_hot = np.zeros((k, 4))
        for j, nuc in enumerate(subseq):
            if nuc in mapping and mapping[nuc] < 4:  # Skip 'N'
                one_hot[j, mapping[nuc]] = 1
        
        # Calculate motif scores for this window
        motif_scores = np.array([
            sum(one_hot[j, b] * motif_pssm['pssm'][j, b] 
                for b in range(4) if one_hot[j, b] > 0)
            for j in range(k)
        ])
        
        # Calculate correlation
        if np.std(window) > 0 and np.std(motif_scores) > 0:
            corr = np.corrcoef(window, motif_scores)[0, 1]
            correlations.append(corr)
        else:
            correlations.append(0)
    
    return np.array(correlations)

def visualize_attribution(seq, attribution, motif_corr=None, save_path=None):
    """
    Visualize attribution scores and optional motif correlation
    
    Args:
        seq: Original sequence
        attribution: Attribution scores
        motif_corr: Optional motif correlation scores
        save_path: Path to save visualization
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot attribution scores
    ax.bar(range(len(attribution)), attribution, alpha=0.7, label="Attribution")
    ax.set_xlabel("Position")
    ax.set_ylabel("Attribution Score")
    
    # Plot motif correlation if available
    if motif_corr is not None:
        # Create second y-axis for correlation
        ax2 = ax.twinx()
        # Pad correlation array to match sequence length
        pad_size = len(attribution) - len(motif_corr)
        padded_corr = np.pad(motif_corr, (0, pad_size), 'constant')
        ax2.plot(range(len(padded_corr)), padded_corr, 'r-', label="Motif Correlation")
        ax2.set_ylabel("Motif Correlation")
        ax2.legend(loc="upper right")
    
    ax.legend(loc="upper left")
    plt.title("Sequence Attribution Analysis")
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def main():
    """
    Главная функция для запуска базового теста интерпретации
    """
    # Тест загрузки модели и config
    model_path = "weight/hyenadna/hyenadna-large-1m-seqlen/weights.ckpt"
    config_path = "weight/hyenadna/hyenadna-large-1m-seqlen/config.json"
    
    if os.path.exists(model_path):
        # Загружаем модель 
        load_model_from_checkpoint(model_path)
    else:
        print(f"Модель не найдена по пути: {model_path}")
    
    if os.path.exists(config_path):
        # Загружаем конфигурацию
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("\nКонфигурация модели:")
        print(json.dumps(config, indent=2))
    else:
        print(f"Конфигурация не найдена по пути: {config_path}")
    
    # Создаем простую тестовую последовательность
    test_seq = "ACGTACGTACGTACGTATATATAAATACGTACGTACGT"
    
    # Создаем директорию для результатов
    output_dir = "interpret_results_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Параметр include_n для one-hot кодирования
    # Проверяем работу с включенным и отключенным N
    for include_n in [True, False]:
        # Токенизируем последовательность
        tokens = tokenize_sequence(test_seq, include_n=include_n)
        print(f"\nТокенизированная последовательность (include_n={include_n}): {tokens.shape}")
        
        # Имитируем атрибуции (вместо реального вычисления с моделью)
        fake_attr = np.random.rand(len(test_seq))
        print(f"Созданы фиктивные атрибуции для демонстрации: {fake_attr.shape}")
        
        # Визуализируем и сохраняем
        save_path = os.path.join(output_dir, f"test_attribution_include_n_{include_n}.png")
        visualize_attribution(test_seq, fake_attr, save_path=save_path)
        print(f"Визуализация сохранена в {save_path}")
    
    # Загружаем пример мотива JASPAR, если доступно
    try:
        motif_id = "MA0108.1"  # TATA-box
        motif_pssm = load_jaspar_motif(motif_id)
        if motif_pssm:
            print(f"Используем демонстрационный мотив TATA-box")
            # Создаем корреляцию с мотивом
            corr = correlate_with_motif(fake_attr, motif_pssm, test_seq)
            save_path = os.path.join(output_dir, "test_attribution_with_motif.png")
            visualize_attribution(test_seq, fake_attr, corr, save_path=save_path)
            print(f"Визуализация с мотивом сохранена в {save_path}")
    except Exception as e:
        print(f"Не удалось загрузить мотив: {e}")

if __name__ == "__main__":
    main() 