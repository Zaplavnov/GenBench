#!/usr/bin/env python
"""
Скрипт для запуска всех тестов проекта GenBench.
"""

import os
import sys
import unittest

# Получаем абсолютный путь к корневой директории проекта
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def run_all_tests():
    """
    Запускает все тесты в директории tests.
    """
    # Находим все тестовые файлы
    test_directory = os.path.dirname(os.path.abspath(__file__))
    test_modules = [
        file[:-3] for file in os.listdir(test_directory)
        if file.startswith('test_') and file.endswith('.py')
    ]
    
    # Загружаем и запускаем тесты
    suite = unittest.TestSuite()
    
    for module_name in test_modules:
        try:
            # Импортируем модуль и добавляем его тесты в сьют
            module = __import__(module_name)
            suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(module))
        except ImportError as e:
            print(f"Ошибка при импорте модуля {module_name}: {e}")
    
    # Запускаем тесты с подробным выводом
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 