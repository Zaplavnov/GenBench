"""
Основной модуль, содержащий классы моделей HyenaDNA.
"""

import torch
import torch.nn as nn

class HyenaModel(nn.Module):
    """
    Базовая модель HyenaDNA для работы с ДНК-последовательностями.
    """
    
    def __init__(self, seq_len=1024, embed_dim=256, n_layers=6):
        """
        Инициализация модели HyenaDNA.
        
        Args:
            seq_len: Максимальная длина последовательности.
            embed_dim: Размерность эмбеддинга.
            n_layers: Количество слоев модели.
        """
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        
        # Здесь будет полная архитектура модели
        self.embedding = nn.Embedding(5, embed_dim)  # 4 нуклеотида + специальный токен
        
    def forward(self, x):
        """
        Прямой проход через модель.
        
        Args:
            x: Входные данные (последовательность ДНК).
            
        Returns:
            Выходные данные модели.
        """
        # Здесь будет полная реализация модели
        x = self.embedding(x)
        return x 