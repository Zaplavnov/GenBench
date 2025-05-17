#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gzip
import random
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq

# Создаем директорию для выходных файлов, если она не существует
os.makedirs('data/probe_data', exist_ok=True)

# Функция для генерации случайных последовательностей ДНК
def generate_random_dna(length=100):
    return ''.join(random.choice(['A', 'C', 'G', 'T']) for _ in range(length))

# Функция для создания датасета мотивов
def create_motif_dataset():
    # Определяем мотив TATA-box
    tata_motif = 'TATAAA'
    
    # Создаем датасет
    sequences = []
    has_motif = []
    
    # Генерируем последовательности с мотивом
    for i in range(50):
        seq_length = random.randint(100, 150)
        seq = generate_random_dna(seq_length)
        
        # Вставляем мотив в случайную позицию
        pos = random.randint(0, seq_length - len(tata_motif))
        seq_with_motif = seq[:pos] + tata_motif + seq[pos+len(tata_motif):]
        
        sequences.append(seq_with_motif)
        has_motif.append(1)
    
    # Генерируем последовательности без мотива
    for i in range(50):
        seq = generate_random_dna(random.randint(100, 150))
        sequences.append(seq)
        has_motif.append(0)
    
    # Создаем DataFrame
    df = pd.DataFrame({
        'sequence': sequences,
        'has_motif': has_motif
    })
    
    # Сохраняем в CSV
    df.to_csv('data/probe_data/motifs.csv', index=False)
    print(f"Создан датасет мотивов: data/probe_data/motifs.csv ({len(df)} записей)")

# Функция для создания датасета хроматиновой доступности
def create_chromatin_dataset():
    # Определим маркеры открытого и закрытого хроматина
    high_gc_content = 0.65  # Высокое содержание GC указывает на открытый хроматин
    
    # Создаем датасет
    sequences = []
    accessibility = []
    
    # Генерируем последовательности с высоким GC (открытый хроматин)
    for i in range(50):
        seq_length = random.randint(100, 150)
        # Больше вероятность выбора G или C
        seq = ''.join(random.choices(['A', 'C', 'G', 'T'], weights=[0.175, 0.325, 0.325, 0.175], k=seq_length))
        sequences.append(seq)
        accessibility.append(1)  # Открытый хроматин
    
    # Генерируем последовательности с низким GC (закрытый хроматин)
    for i in range(50):
        seq_length = random.randint(100, 150)
        # Больше вероятность выбора A или T
        seq = ''.join(random.choices(['A', 'C', 'G', 'T'], weights=[0.325, 0.175, 0.175, 0.325], k=seq_length))
        sequences.append(seq)
        accessibility.append(0)  # Закрытый хроматин
    
    # Создаем DataFrame
    df = pd.DataFrame({
        'sequence': sequences,
        'accessibility': accessibility
    })
    
    # Сохраняем в CSV
    df.to_csv('data/probe_data/chromatin.csv', index=False)
    print(f"Создан датасет хроматиновой доступности: data/probe_data/chromatin.csv ({len(df)} записей)")

# Функция для создания датасета экспрессии генов
def create_expression_dataset():
    # Определим последовательности промоторов (упрощенно)
    promoter_motifs = ['TATAAA', 'CAAT', 'GC']
    
    # Создаем датасет
    sequences = []
    expression_levels = []
    
    # Генерируем последовательности с разным уровнем экспрессии
    for i in range(100):
        seq_length = random.randint(100, 150)
        seq = generate_random_dna(seq_length)
        
        # Определяем уровень экспрессии на основе случайных факторов
        expression = random.uniform(0, 10)
        
        # Добавляем больше мотивов промоторов для высокоэкспрессируемых генов
        if expression > 5:
            num_motifs = random.randint(2, 4)
            for _ in range(num_motifs):
                motif = random.choice(promoter_motifs)
                pos = random.randint(0, seq_length - len(motif))
                seq = seq[:pos] + motif + seq[pos+len(motif):]
        
        sequences.append(seq)
        expression_levels.append(expression)
    
    # Создаем DataFrame
    df = pd.DataFrame({
        'sequence': sequences,
        'expression_level': expression_levels
    })
    
    # Сохраняем в CSV
    df.to_csv('data/probe_data/expression.csv', index=False)
    print(f"Создан датасет экспрессии генов: data/probe_data/expression.csv ({len(df)} записей)")

# Функция для создания датасета метилирования ДНК
def create_methylation_dataset():
    # Создаем датасет
    sequences = []
    methylation_levels = []
    
    # Генерируем последовательности с разным уровнем метилирования
    for i in range(100):
        seq_length = random.randint(100, 150)
        # Выбираем количество CpG сайтов
        cpg_count = random.randint(0, 10)
        
        # Генерируем базовую последовательность
        seq = generate_random_dna(seq_length)
        
        # Добавляем CpG сайты
        for _ in range(cpg_count):
            pos = random.randint(0, seq_length - 2)
            seq = seq[:pos] + 'CG' + seq[pos+2:]
        
        # Определяем уровень метилирования (на основе количества CpG)
        methylation = random.uniform(0, 1) if cpg_count > 0 else 0
        
        sequences.append(seq)
        methylation_levels.append(methylation)
    
    # Создаем DataFrame
    df = pd.DataFrame({
        'sequence': sequences,
        'methylation_level': methylation_levels
    })
    
    # Сохраняем в CSV
    df.to_csv('data/probe_data/methylation.csv', index=False)
    print(f"Создан датасет метилирования ДНК: data/probe_data/methylation.csv ({len(df)} записей)")

# Функция для создания датасета сплайсинга
def create_splicing_dataset():
    # Определяем консенсусные последовательности сайтов сплайсинга
    donor_motif = 'GTAAGT'  # 5' сайт сплайсинга (донор)
    acceptor_motif = 'TTTTCAG'  # 3' сайт сплайсинга (акцептор)
    
    # Создаем датасет
    sequences = []
    splicing_types = []
    
    # Генерируем последовательности с донорным сайтом сплайсинга
    for i in range(30):
        seq_length = random.randint(100, 150)
        seq = generate_random_dna(seq_length)
        
        # Вставляем донорный мотив
        pos = random.randint(20, seq_length - 20 - len(donor_motif))
        seq = seq[:pos] + donor_motif + seq[pos+len(donor_motif):]
        
        sequences.append(seq)
        splicing_types.append('donor')
    
    # Генерируем последовательности с акцепторным сайтом сплайсинга
    for i in range(30):
        seq_length = random.randint(100, 150)
        seq = generate_random_dna(seq_length)
        
        # Вставляем акцепторный мотив
        pos = random.randint(20, seq_length - 20 - len(acceptor_motif))
        seq = seq[:pos] + acceptor_motif + seq[pos+len(acceptor_motif):]
        
        sequences.append(seq)
        splicing_types.append('acceptor')
    
    # Генерируем последовательности без сайтов сплайсинга
    for i in range(40):
        seq = generate_random_dna(random.randint(100, 150))
        sequences.append(seq)
        splicing_types.append('none')
    
    # Создаем DataFrame
    df = pd.DataFrame({
        'sequence': sequences,
        'splicing_type': splicing_types
    })
    
    # Сохраняем в CSV
    df.to_csv('data/probe_data/splicing.csv', index=False)
    print(f"Создан датасет сплайсинга: data/probe_data/splicing.csv ({len(df)} записей)")

if __name__ == "__main__":
    print("Подготовка данных для задач пробинга...")
    create_motif_dataset()
    create_chromatin_dataset()
    create_expression_dataset()
    create_methylation_dataset()
    create_splicing_dataset()
    print("Подготовка данных завершена успешно!") 