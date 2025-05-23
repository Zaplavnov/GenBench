# Примеры использования GenBench

В этой директории представлены примеры использования функциональности GenBench для работы с геномными моделями.

## Структура директории

- `basic_usage/` - базовые примеры использования основных компонентов
  - `hyenadna_example.py` - пример загрузки и использования модели HyenaDNA
  - `probe_example.py` - пример использования пробинговых задач
  - `interpret_example.py` - пример интерпретации моделей с помощью Captum

## Запуск примеров

Все примеры можно запускать из корневой директории проекта:

```bash
# Базовый пример использования HyenaDNA
python examples/basic_usage/hyenadna_example.py

# Пример пробинговых задач
python examples/basic_usage/probe_example.py

# Пример интерпретации с Captum
python examples/basic_usage/interpret_example.py
```

## Предварительные требования

Для запуска примеров необходимо:

1. Установить зависимости проекта: `pip install -r requirements.txt`
2. Загрузить предобученные модели HyenaDNA: `python download_models.py --model_type tiny-1k-seqlen`
3. Для примера интерпретации установить дополнительные зависимости: `pip install captum`

## Примечания

- Примеры используют модель HyenaDNA small, которую можно заменить на другую, изменив параметры `model_path` и `config_path`.
- Для работы с реальными данными, замените примеры последовательностей ДНК на ваши собственные.
- Пробинговые задачи используют синтетические данные для демонстрации, но могут быть адаптированы для работы с реальными данными. 