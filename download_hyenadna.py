#!/usr/bin/env python

import os
import sys
from huggingface_hub import hf_hub_download
import json

def main():
    """
    Скачивает модель HyenaDNA с HuggingFace и сохраняет ее в директории weight/hyenadna
    """
    
    # Параметры модели
    repo_id = "LongSafari/hyenadna-large-1m-seqlen"
    output_dir = "weight/hyenadna/hyenadna-large-1m-seqlen"
    
    # Создаем директорию для сохранения модели
    os.makedirs(output_dir, exist_ok=True)
    
    # Список файлов для скачивания
    files_to_download = {
        "weights.ckpt": "weights.ckpt",
        "config.json": "config.json",
    }
    
    print(f"Скачиваем модель {repo_id} в {output_dir}")
    
    # Скачиваем каждый файл
    for filename, output_filename in files_to_download.items():
        try:
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=output_dir,
                local_dir_use_symlinks=False
            )
            print(f"Успешно скачан файл {filename} в {file_path}")
        except Exception as e:
            print(f"Ошибка при скачивании {filename}: {e}")
            
    print("Завершено скачивание модели.")
    
if __name__ == "__main__":
    main() 