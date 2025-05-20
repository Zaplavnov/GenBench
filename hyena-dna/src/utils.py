"""
Вспомогательные функции для работы с моделями HyenaDNA.
"""

import numpy as np

def process_dna_sequence(sequence, max_length=1024):
    """
    Обработка ДНК-последовательности.
    
    Args:
        sequence: ДНК-последовательность в виде строки.
        max_length: Максимальная длина последовательности.
        
    Returns:
        Обработанная последовательность.
    """
    # Преобразование нуклеотидов в числовые значения
    # A:0, C:1, G:2, T:3, N:4
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    
    # Преобразование строки в последовательность чисел
    processed = [mapping.get(nucleotide.upper(), 4) for nucleotide in sequence]
    
    # Обрезка или дополнение до max_length
    if len(processed) > max_length:
        processed = processed[:max_length]
    else:
        processed += [4] * (max_length - len(processed))
    
    return np.array(processed)

def one_hot_encode(sequence, max_length=1024):
    """
    One-hot кодирование ДНК-последовательности.
    
    Args:
        sequence: ДНК-последовательность в виде строки.
        max_length: Максимальная длина последовательности.
        
    Returns:
        One-hot кодированная последовательность.
    """
    # One-hot кодирование для 4 нуклеотидов + N
    mapping = {'A': [1, 0, 0, 0, 0], 
               'C': [0, 1, 0, 0, 0], 
               'G': [0, 0, 1, 0, 0], 
               'T': [0, 0, 0, 1, 0], 
               'N': [0, 0, 0, 0, 1]}
    
    # Преобразование строки в one-hot кодирование
    encoded = [mapping.get(nucleotide.upper(), [0, 0, 0, 0, 1]) for nucleotide in sequence]
    
    # Обрезка или дополнение до max_length
    if len(encoded) > max_length:
        encoded = encoded[:max_length]
    else:
        encoded += [[0, 0, 0, 0, 1]] * (max_length - len(encoded))
    
    return np.array(encoded) 