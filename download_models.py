#!/usr/bin/env python
"""
Универсальный скрипт для загрузки моделей HyenaDNA различных размеров и форматов.
Заменяет отдельные скрипты download_hyenadna.py и download_hyenadna_models.py.
"""

import os
import argparse
import requests
import json
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import hf_hub_download


def download_file(url, output_path):
    """
    Загружает файл с указанного URL и сохраняет по указанному пути с индикатором прогресса
    
    Args:
        url: URL для скачивания
        output_path: путь для сохранения файла
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Получаем размер файла, если он указан в заголовках
    total_size = int(response.headers.get('content-length', 0))
    
    # Создаем директорию, если она не существует
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        if total_size == 0:
            f.write(response.content)
        else:
            # Используем tqdm для отображения прогресса
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(output_path)) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))


def download_from_hf_hub(repo_id, model_name, output_dir, use_hf_format=False):
    """
    Загружает модель с Hugging Face Hub с использованием API huggingface_hub
    
    Args:
        repo_id: ID репозитория на Hugging Face Hub
        model_name: имя модели (для папки назначения)
        output_dir: директория для сохранения модели
        use_hf_format: использовать ли HuggingFace формат модели
        
    Returns:
        dict: пути к загруженным файлам
    """
    # Формируем путь для сохранения модели
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Загружаем модель {repo_id} в {model_dir}")
    
    # Файлы для загрузки в зависимости от формата
    if use_hf_format:
        files_to_download = {
            "config.json": "config.json",
            "pytorch_model.bin": "pytorch_model.bin",
            "tokenizer_config.json": "tokenizer_config.json",
            "special_tokens_map.json": "special_tokens_map.json"
        }
    else:
        files_to_download = {
            "weights.ckpt": "weights.ckpt",
            "config.json": "config.json"
        }
    
    file_paths = {}
    
    # Загружаем каждый файл
    for filename, output_filename in files_to_download.items():
        output_path = os.path.join(model_dir, output_filename)
        
        # Пропускаем, если файл уже существует
        if os.path.exists(output_path):
            print(f"Файл {output_filename} уже существует. Пропускаем...")
            file_paths[filename.split('.')[0]] = output_path
            continue
        
        try:
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=model_dir,
                local_dir_use_symlinks=False
            )
            print(f"Успешно загружен файл {filename} в {file_path}")
            file_paths[filename.split('.')[0]] = file_path
        except Exception as e:
            print(f"Ошибка при загрузке {filename}: {e}")
            # Для больших файлов попробуем альтернативный метод
            if filename in ["weights.ckpt", "pytorch_model.bin"]:
                try:
                    direct_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
                    print(f"Пытаемся загрузить напрямую с URL: {direct_url}")
                    download_file(direct_url, output_path)
                    file_paths[filename.split('.')[0]] = output_path
                    print(f"Успешно загружен файл {filename} напрямую")
                except Exception as e2:
                    print(f"Ошибка при загрузке напрямую: {e2}")
    
    # Проверяем, что загружены необходимые файлы
    if use_hf_format:
        if not all(k in file_paths for k in ["config", "pytorch_model"]):
            print("Не удалось загрузить необходимые файлы для формата HuggingFace")
            return None
    else:
        if not all(k in file_paths for k in ["weights", "config"]):
            print("Не удалось загрузить необходимые файлы для оригинального формата")
            return None
    
    print(f"Модель {model_name} успешно загружена в {model_dir}")
    return file_paths


def main():
    parser = argparse.ArgumentParser(description="Загрузка предобученных моделей HyenaDNA")
    parser.add_argument("--model_type", type=str, default="tiny-1k-seqlen", 
                        choices=["tiny-1k-seqlen", "tiny-1k-seqlen-d256", "tiny-16k-seqlen-d128",
                                 "small-32k-seqlen", "medium-160k-seqlen", "medium-450k-seqlen", 
                                 "large-1m-seqlen"], 
                        help="Версия модели HyenaDNA для загрузки")
    parser.add_argument("--output_dir", type=str, default="models", 
                        help="Директория для сохранения моделей")
    parser.add_argument("--use_hf_format", action="store_true", 
                        help="Использовать формат HuggingFace вместо оригинального")
    parser.add_argument("--legacy_output_dir", action="store_true",
                        help="Использовать старый формат пути (weight/hyenadna)")
    
    args = parser.parse_args()
    
    # Определяем имя репозитория на HuggingFace
    model_name = f"hyenadna-{args.model_type}"
    if args.use_hf_format:
        model_name = f"{model_name}-hf"
    
    repo_id = f"LongSafari/{model_name}"
    
    # Определяем выходную директорию
    output_dir = args.output_dir
    if args.legacy_output_dir:
        output_dir = "weight/hyenadna"
    
    # Загружаем модель
    file_paths = download_from_hf_hub(repo_id, model_name, output_dir, args.use_hf_format)
    
    if file_paths:
        print("\nПути к загруженным файлам:")
        for file_type, path in file_paths.items():
            print(f"  {file_type}: {path}")
        
        if "weights" in file_paths:
            print("\nДля использования модели в скрипте интерпретации используйте (оригинальный формат):")
            print(f"python interpret_hyenadna_captum.py --weights_path {file_paths['weights']} --config_path {file_paths['config']}")
        elif "pytorch_model" in file_paths:
            print("\nДля использования модели в скрипте интерпретации используйте (формат HuggingFace):")
            print(f"python interpret_hyenadna_captum.py --model_path {os.path.dirname(file_paths['pytorch_model'])} --hf_format True")


if __name__ == "__main__":
    main() 