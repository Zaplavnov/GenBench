import os
import sys
import json
import torch
import numpy as np

# Добавляем репозиторий hyena-dna в путь
hyena_dna_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "hyena-dna")
sys.path.append(hyena_dna_path)

# Импортируем необходимые модули из hyena-dna
try:
    from hyena_dna.model.sequence_model import SequenceModel
except ImportError:
    print("Не удалось импортировать SequenceModel из hyena-dna.")
    print(f"Проверьте, что репозиторий hyena-dna находится по пути: {hyena_dna_path}")


def load_hyenadna_model(weights_path, config_path=None):
    """
    Загружает модель HyenaDNA из весов и конфига
    
    Args:
        weights_path: Путь к весам модели (.ckpt файл)
        config_path: Путь к файлу конфигурации (config.json)
        
    Returns:
        Загруженная модель HyenaDNA
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Файл с весами не найден: {weights_path}")
    
    # Если путь к конфигу не указан, пытаемся определить его автоматически
    if config_path is None:
        config_path = os.path.join(os.path.dirname(weights_path), "config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
    
    # Загружаем конфигурацию модели
    with open(config_path, "r") as f:
        config = json.load(f)
    
    print(f"Загружена конфигурация модели: {config_path}")
    
    # Создаем экземпляр модели на основе конфигурации
    model_config = {"d_model": config["d_model"],
                    "n_layer": config["n_layer"],
                    "d_inner": config["d_inner"],
                    "vocab_size": config["vocab_size"],
                    "resid_dropout": config["resid_dropout"],
                    "embed_dropout": config["embed_dropout"],
                    "layer": config["layer"]}
    
    model = SequenceModel(model_config)
    print(f"Создана модель HyenaDNA с параметрами:\n- d_model: {config['d_model']}\n- n_layer: {config['n_layer']}")
    
    # Загружаем веса модели
    try:
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print(f"Загружены веса из state_dict ({len(state_dict)} параметров)")
            
            # Удаляем префикс "model." из ключей state_dict если он есть
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            
            # Загружаем веса в модель
            model.load_state_dict(new_state_dict)
        else:
            print("Checkpoint не содержит state_dict. Пытаемся загрузить напрямую...")
            model.load_state_dict(checkpoint)
        
        print(f"Успешно загружены веса из {weights_path}")
    except Exception as e:
        print(f"Ошибка при загрузке весов: {e}")
        raise
    
    # Переводим модель в режим оценки
    model.eval()
    
    return model


def extract_embeddings_hyenadna(model, sequences, device='cuda', batch_size=32, layer_idx=-1):
    """
    Извлекает эмбеддинги из модели HyenaDNA
    
    Args:
        model: Модель HyenaDNA
        sequences: Список последовательностей ДНК
        device: Устройство для вычислений ('cuda' или 'cpu')
        batch_size: Размер батча для инференса
        layer_idx: Индекс слоя, из которого извлекаются эмбеддинги (-1 для последнего)
        
    Returns:
        np.ndarray: Массив эмбеддингов размера [num_sequences, d_model]
    """
    device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
    model = model.to(device)
    model.eval()
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            
            # Токенизируем последовательности
            # HyenaDNA принимает последовательности в виде строк
            inputs = {}
            inputs["sequence"] = batch_seqs
            
            # Делаем инференс
            outputs = model(inputs, return_all_hiddens=True)
            
            # Получаем скрытые состояния
            hidden_states = outputs["hidden_states"]
            
            # Берем эмбеддинги из указанного слоя
            embeddings = hidden_states[layer_idx]
            
            # Среднее по всей последовательности для получения вектора фиксированной длины
            seq_embeddings = embeddings.mean(dim=1).cpu().numpy()
            all_embeddings.append(seq_embeddings)
    
    return np.vstack(all_embeddings) 