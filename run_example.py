#!/usr/bin/env python

import os
import sys
import subprocess

def main():
    """
    Запуск интерпретации модели HyenaDNA на примере demo_human_or_worm
    """
    
    # Путь к модели
    model_path = "weight/hyenadna/hyenadna-large-1m-seqlen"
    
    # Проверяем наличие модели
    if not os.path.exists(model_path):
        print(f"Ошибка: Модель не найдена в {model_path}")
        print("Сначала скачайте модель, запустив python download_hyenadna.py")
        return
    
    # Команда для запуска интерпретации
    command = [
        "python", "interpret.py",
        "experiment=hg38/genomic_benchmark_hyena",
        "dataset.dataset_name=demo_human_or_worm",
        f"train.pretrained_model_path={model_path}",
        "interpret.attribution_method=ig",
        "interpret.num_samples=3",
        "interpret.output_dir=interpret_results_demo",
        "wandb.mode=disabled"
    ]
    
    print("Запускаем интерпретацию модели HyenaDNA...")
    print(f"Команда: {' '.join(command)}")
    
    # Запускаем процесс
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Выводим результаты в реальном времени
    for line in process.stdout:
        print(line, end='')
    
    # Ждем завершения процесса
    process.wait()
    
    # Проверяем результат
    if process.returncode == 0:
        print("Интерпретация успешно завершена!")
        print(f"Результаты сохранены в директории interpret_results_demo")
    else:
        print(f"Ошибка при выполнении интерпретации. Код возврата: {process.returncode}")
        for line in process.stderr:
            print(line, end='')
    
if __name__ == "__main__":
    main() 