#!/usr/bin/env python
"""
Скрипт для стандартизации импортов и форматирования кода в репозитории GenBench.
Использует isort для сортировки импортов и black для форматирования кода.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """
    Запускает команду и выводит описание результата.
    
    Args:
        cmd: Список аргументов команды.
        description: Описание действия.
    
    Returns:
        bool: True, если команда выполнена успешно, иначе False.
    """
    print(f"Выполняется: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"{description} выполнено успешно.")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при {description.lower()}:")
        print(e.stderr)
        return False


def format_directory(directory_path, python_files=None):
    """
    Форматирует все Python-файлы в указанной директории.
    
    Args:
        directory_path: Путь к директории.
        python_files: Список Python-файлов для форматирования (если None, форматируются все).
    
    Returns:
        bool: True, если все форматирование выполнено успешно, иначе False.
    """
    directory = Path(directory_path)
    
    if python_files is None:
        # Ищем все Python-файлы рекурсивно
        python_files = list(directory.glob("**/*.py"))
    
    if not python_files:
        print(f"В директории {directory} не найдено Python-файлов.")
        return True
    
    print(f"Найдено {len(python_files)} Python-файлов в {directory}.")
    
    # Преобразуем пути к строкам
    python_files_str = [str(file) for file in python_files]
    
    # Форматируем с помощью isort
    isort_success = run_command(
        ["isort"] + python_files_str,
        "Сортировка импортов (isort)"
    )
    
    # Форматируем с помощью black
    black_success = run_command(
        ["black"] + python_files_str,
        "Форматирование кода (black)"
    )
    
    # Проверяем на соответствие стилю с помощью flake8
    flake8_success = run_command(
        ["flake8"] + python_files_str,
        "Проверка стиля (flake8)"
    )
    
    return isort_success and black_success and flake8_success


def main():
    """
    Основная функция скрипта.
    """
    # Получаем путь к корневой директории проекта
    root_dir = Path(__file__).parent.parent.absolute()
    
    # Директории, которые нужно форматировать
    directories_to_format = [
        root_dir / "src",
        root_dir / "examples",
        root_dir / "tests",
        root_dir / "scripts",
    ]
    
    # Отдельные файлы в корневой директории
    root_python_files = list((root_dir).glob("*.py"))
    
    # Форматируем каждую директорию
    all_success = True
    for directory in directories_to_format:
        if directory.exists():
            print(f"\nФорматирование директории: {directory}\n" + "-" * 50)
            success = format_directory(directory)
            all_success = all_success and success
        else:
            print(f"Директория {directory} не существует.")
    
    # Форматируем файлы в корневой директории
    if root_python_files:
        print(f"\nФорматирование файлов в корневой директории\n" + "-" * 50)
        success = format_directory(root_dir, root_python_files)
        all_success = all_success and success
    
    if all_success:
        print("\nВсе операции форматирования выполнены успешно.")
        return 0
    else:
        print("\nНекоторые операции форматирования завершились с ошибками.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 