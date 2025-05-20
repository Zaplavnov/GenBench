#!/usr/bin/env python
"""
Тесты для проверки работы токенизатора ДНК.
"""

import os
import sys
import unittest
import numpy as np

# Добавляем корневую директорию в sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем необходимые модули из проекта GenBench
from src.utils.tokenizer_dna import DNATokenizer
from hyena_dna.utils import process_dna_sequence, one_hot_encode


class TestDNATokenization(unittest.TestCase):
    """
    Тесты для проверки работы токенизатора ДНК.
    """
    
    def setUp(self):
        """
        Инициализация перед каждым тестом.
        """
        self.tokenizer = DNATokenizer()
        self.dna_sequence = "ACGTACGTACGT"
        
    def test_tokenizer_basic(self):
        """
        Проверка базовой токенизации ДНК-последовательности.
        """
        tokens = self.tokenizer.tokenize(self.dna_sequence)
        self.assertEqual(len(tokens), len(self.dna_sequence))
        self.assertEqual(tokens[0], 0)  # 'A' -> 0
        self.assertEqual(tokens[1], 1)  # 'C' -> 1
        self.assertEqual(tokens[2], 2)  # 'G' -> 2
        self.assertEqual(tokens[3], 3)  # 'T' -> 3
        
    def test_tokenizer_unknown(self):
        """
        Проверка токенизации с неизвестными символами.
        """
        unknown_sequence = "ACGTNX"
        tokens = self.tokenizer.tokenize(unknown_sequence)
        self.assertEqual(len(tokens), len(unknown_sequence))
        self.assertEqual(tokens[4], 4)  # 'N' -> 4 (неизвестный нуклеотид)
        self.assertEqual(tokens[5], 4)  # 'X' -> 4 (неизвестный символ)
        
    def test_one_hot_encoding(self):
        """
        Проверка one-hot кодирования ДНК-последовательности.
        """
        encoded = one_hot_encode(self.dna_sequence, max_length=len(self.dna_sequence))
        self.assertEqual(encoded.shape, (len(self.dna_sequence), 5))
        
        # Проверяем, что 'A' кодируется как [1, 0, 0, 0, 0]
        np.testing.assert_array_equal(encoded[0], [1, 0, 0, 0, 0])
        
        # Проверяем, что 'C' кодируется как [0, 1, 0, 0, 0]
        np.testing.assert_array_equal(encoded[1], [0, 1, 0, 0, 0])
        
        # Проверяем, что 'G' кодируется как [0, 0, 1, 0, 0]
        np.testing.assert_array_equal(encoded[2], [0, 0, 1, 0, 0])
        
        # Проверяем, что 'T' кодируется как [0, 0, 0, 1, 0]
        np.testing.assert_array_equal(encoded[3], [0, 0, 0, 1, 0])
        
    def test_process_dna_sequence(self):
        """
        Проверка обработки ДНК-последовательности.
        """
        processed = process_dna_sequence(self.dna_sequence, max_length=len(self.dna_sequence))
        self.assertEqual(len(processed), len(self.dna_sequence))
        self.assertEqual(processed[0], 0)  # 'A' -> 0
        self.assertEqual(processed[1], 1)  # 'C' -> 1
        self.assertEqual(processed[2], 2)  # 'G' -> 2
        self.assertEqual(processed[3], 3)  # 'T' -> 3
        
    def test_padding(self):
        """
        Проверка дополнения последовательности до заданной длины.
        """
        short_sequence = "ACG"
        max_length = 10
        
        # Проверка для простой токенизации
        processed = process_dna_sequence(short_sequence, max_length=max_length)
        self.assertEqual(len(processed), max_length)
        self.assertEqual(processed[0], 0)  # 'A' -> 0
        self.assertEqual(processed[1], 1)  # 'C' -> 1
        self.assertEqual(processed[2], 2)  # 'G' -> 2
        self.assertEqual(processed[3], 4)  # Padding -> 4
        
        # Проверка для one-hot кодирования
        encoded = one_hot_encode(short_sequence, max_length=max_length)
        self.assertEqual(encoded.shape, (max_length, 5))
        np.testing.assert_array_equal(encoded[0], [1, 0, 0, 0, 0])  # 'A'
        np.testing.assert_array_equal(encoded[1], [0, 1, 0, 0, 0])  # 'C'
        np.testing.assert_array_equal(encoded[2], [0, 0, 1, 0, 0])  # 'G'
        np.testing.assert_array_equal(encoded[3], [0, 0, 0, 0, 1])  # Padding
        

if __name__ == "__main__":
    unittest.main() 