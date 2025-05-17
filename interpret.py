#!/usr/bin/env python

import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

# Добавляем корневой каталог в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Импортируем модуль интерпретации
from src.utils.interpret import run_interpretation

@hydra.main(config_path="configs", config_name="config.yaml")
def main(config: DictConfig):
    """
    Main entry point for model interpretation
    
    Usage:
        python interpret.py dataset.dataset_name=<dataset_name> train.pretrained_model_path=<path_to_model> ...
        
    Optional:
        interpret.include_n=<true|false> - Whether to include N as a separate channel in one-hot encoding
    """
    # Проверяем наличие необходимых параметров
    if not hasattr(config, "interpret"):
        print("Error: Config does not contain 'interpret' section. Make sure it's properly set up.")
        sys.exit(1)
        
    if not hasattr(config, "ckpt_path") and not hasattr(config.interpret, "ckpt_path"):
        if hasattr(config.train, "pretrained_model_path") and config.train.pretrained_model_path:
            config.interpret.ckpt_path = config.train.pretrained_model_path
        else:
            print("Error: No checkpoint path specified. Use 'interpret.ckpt_path=<path>' or 'train.pretrained_model_path=<path>'")
            sys.exit(1)
    
    # Установка значения по умолчанию для include_n, если не указано
    if not hasattr(config.interpret, "include_n"):
        config.interpret.include_n = True
            
    # Запускаем интерпретацию
    run_interpretation(config)
    
if __name__ == "__main__":
    main() 