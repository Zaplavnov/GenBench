#!/usr/bin/env python

import os
import requests
import json
import argparse
from tqdm import tqdm
from pathlib import Path


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


def download_hyenadna_model(model_name, output_dir="models", use_hf_format=False):
    """
    Загружает предобученную модель HyenaDNA с Hugging Face
    
    Args:
        model_name: имя модели (например, "tiny-1k-seqlen", "small-32k-seqlen" и т.д.)
        output_dir: директория для сохранения модели
        use_hf_format: использовать ли HuggingFace формат модели (по умолчанию False - оригинальный формат)
        
    Returns:
        dict: пути к загруженным файлам
    """
    # Добавляем -hf к имени модели, если нужен HuggingFace формат
    full_model_name = model_name
    if use_hf_format:
        full_model_name = f"{model_name}-hf"
    
    base_url = f"https://huggingface.co/LongSafari/hyenadna-{full_model_name}/resolve/main"
    
    # Создаем директорию для загрузки
    model_dir = os.path.join(output_dir, f"hyenadna-{full_model_name}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Файлы для загрузки в зависимости от формата
    if use_hf_format:
        # Для моделей HuggingFace используем формат HF
        files = {
            "config": f"{base_url}/config.json",
            "model": f"{base_url}/pytorch_model.bin",
            "tokenizer_config": f"{base_url}/tokenizer_config.json",
            "special_tokens_map": f"{base_url}/special_tokens_map.json",
        }
        
        file_paths = {
            "config": os.path.join(model_dir, "config.json"),
            "model": os.path.join(model_dir, "pytorch_model.bin"),
            "tokenizer_config": os.path.join(model_dir, "tokenizer_config.json"),
            "special_tokens_map": os.path.join(model_dir, "special_tokens_map.json"),
        }
    else:
        # Для оригинальных моделей используем формат с weights.ckpt
        files = {
            "config": f"{base_url}/config.json",
            "weights": f"{base_url}/weights.ckpt",
        }
        
        file_paths = {
            "config": os.path.join(model_dir, "config.json"),
            "weights": os.path.join(model_dir, "weights.ckpt"),
        }
    
    # Загружаем файлы
    print(f"Загрузка модели hyenadna-{full_model_name}...")
    
    success_count = 0
    for file_type, url in files.items():
        output_path = file_paths[file_type]
        
        # Пропускаем, если файл уже существует
        if os.path.exists(output_path):
            print(f"Файл {os.path.basename(output_path)} уже существует. Пропускаем...")
            success_count += 1
            continue
        
        print(f"Загрузка {file_type}...")
        try:
            download_file(url, output_path)
            success_count += 1
        except Exception as e:
            print(f"Ошибка при загрузке {file_type}: {e}")
            # Если это критичный файл, продолжаем попытки
            if file_type in ["config", "weights", "model"]:
                # Пытаемся загрузить из LFS
                try:
                    lfs_url = f"{base_url}/resolve/main/{os.path.basename(output_path)}"
                    print(f"Пытаемся загрузить из LFS: {lfs_url}")
                    download_file(lfs_url, output_path)
                    success_count += 1
                except Exception as e2:
                    print(f"Ошибка при загрузке из LFS: {e2}")
    
    if success_count < 2:  # Должны загрузиться хотя бы config и weights/model
        print("Не удалось загрузить необходимые файлы модели")
        return None
    
    print(f"Модель hyenadna-{full_model_name} успешно загружена в {model_dir}")
    return file_paths


def main():
    parser = argparse.ArgumentParser(description="Загрузка предобученных моделей HyenaDNA")
    parser.add_argument("--model_name", type=str, default="tiny-1k-seqlen", 
                        choices=["tiny-1k-seqlen", "tiny-1k-seqlen-d256", "tiny-16k-seqlen-d128",
                                 "small-32k-seqlen", "medium-160k-seqlen", "medium-450k-seqlen", 
                                 "large-1m-seqlen"], 
                        help="Версия модели HyenaDNA для загрузки")
    parser.add_argument("--output_dir", type=str, default="models", help="Директория для сохранения моделей")
    parser.add_argument("--use_hf_format", action="store_true", help="Использовать формат HuggingFace вместо оригинального")
    
    args = parser.parse_args()
    
    # Загружаем модель
    model_paths = download_hyenadna_model(args.model_name, args.output_dir, args.use_hf_format)
    
    if model_paths:
        print("Пути к загруженным файлам:")
        for file_type, path in model_paths.items():
            print(f"  {file_type}: {path}")
        
        if "weights" in model_paths:
            print("\nДля использования модели в скрипте интерпретации используйте (оригинальный формат):")
            print(f"python interpret_hyenadna_captum.py --weights_path {model_paths['weights']} --config_path {model_paths['config']}")
        else:
            print("\nДля использования модели в скрипте интерпретации используйте (формат HuggingFace):")
            print(f"python interpret_hyenadna_captum.py --model_path {os.path.dirname(model_paths['model'])} --hf_format True")


if __name__ == "__main__":
    main() 