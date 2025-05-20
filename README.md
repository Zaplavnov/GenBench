# HyenaDNA

Компонент для работы с ДНК-последовательностями.

Это часть проекта GenBench.

[NIPS 2024] GenBench: A Comprehensive Benchmark of genomic foundation models
<p align="center" width="100%">
  <img src='assets/genbench.webp' width="100%">
</p>


## Introduction

GenBench is a comprehensive benchmark for evaluating genomic foundation model, encompassing a broad spectrum of methods and diverse tasks, ranging from predicting gene location and function, identifying regulatory elements, and studying species evolution. GenBench offers a modular and extensible framework, excelling in user-friendliness, organization, and comprehensiveness. The codebase is organized into three abstracted layers, namely the core layer, algorithm layer, and user interface layer, arranged from the bottom to the top.

<p align="center" width="100%">
  <img src='assets/architecture.png' width="90%">
</p>


<p align="right">(<a href="#top">back to top</a>)</p>

## Overview



</details>

<details open>
<summary>Code Structures</summary>

- `GenBench/configs/` - Конфигурационные файлы для бенчмарка и моделей:
  - `callbacks/` - Конфигурации обратных вызовов для тренировки
  - `dataset/` - Конфигурации для различных геномных датасетов
  - `experiment/` - Конфигурации для конкретных экспериментов
  - `model/` - Конфигурации моделей и слоев
  - `pipeline/` - Конфигурации обучающих пайплайнов
  - `scheduler/` - Конфигурации оптимизаторов и планировщиков скорости обучения
  - `task/` - Конфигурации для различных задач геномной биоинформатики
  - `trainer/` - Общие настройки тренировочных процессов

- `GenBench/data/` - Директория для данных и датасетов:
  - Включает промоторы, последовательности ChIP-seq и другие геномные данные
  - Файлы должны быть загружены или сгенерированы перед запуском экспериментов

- `GenBench/notebook/` - Jupyter-ноутбуки для анализа и визуализации:
  - Ноутбуки для визуализации результатов (структура генома, активность энхансеров)
  - Ноутбуки для анализа производительности и сравнения моделей
  - Ноутбуки для обработки данных и кластеризации

- `GenBench/src/` - Исходный код основного функционала:
  - `callbacks/` - Имплементации колбэков для тренировки
  - `dataloaders/` - Загрузчики данных для разных форматов геномных данных
  - `models/` - Имплементации моделей и архитектур (HyenaDNA, DNABERT, и др.)
  - `ops/` - Низкоуровневые операции для эффективных вычислений
  - `tasks/` - Реализации задач и метрик оценки
  - `utils/` - Вспомогательные функции и утилиты

- `GenBench/hyena_dna/` - Модуль HyenaDNA для работы с ДНК-последовательностями:
  - Реализация архитектуры HyenaDNA
  - Специализированные функции для ДНК-анализа

- `GenBench/weight/` - Веса предобученных моделей (доступны после загрузки)

- `GenBench/experiment/` - Скрипты для управления экспериментами:
  - Скрипты для различных задач (предсказание промоторов, сплайсинг и др.)
  - `.sh` файлы для запуска экспериментов с разными моделями

- `GenBench/docs/` - Документация по проекту

- `GenBench/examples/` - Примеры запуска и использования моделей

- `GenBench/scripts/` - Вспомогательные скрипты для подготовки данных

- Корневые файлы:
  - `download_models.py` - Скрипт для загрузки предобученных моделей
  - `interpret.py` и `interpret_hyenadna_captum.py` - Инструменты для интерпретации моделей
  - `setup.py` и `environment.yml` - Файлы для установки и настройки окружения
  - `train.py` - Основной скрипт для запуска тренировки моделей
  - `visualize_results.py` - Скрипт для визуализации результатов экспериментов

</details>



## Installation

This project has provided an environment setting file of conda, users can easily reproduce the environment by the following commands:
```shell

cd GenBench
conda env create -f environment.yml
conda activate OpenGenome
python setup.py develop
```

<!-- <details close>
<summary>Dependencies</summary>

* argparse
* dask
* decord
* fvcore
* hickle
* lpips
* matplotlib
* netcdf4
* numpy
* opencv-python
* packaging
* pandas
* python<=3.10.8
* scikit-image
* scikit-learn
* torch
* timm
* tqdm
* xarray==0.19.0
</details>

Please refer to [install.md](docs/en/install.md) for more detailed instructions. -->

## Getting Started

Here is an example of single GPU non-distributed training HyenaDNA on demo_human_or_worm dataset.
```shell
bash tools/prepare_data/download_mmnist.sh
python train.py -m train experiment=hg38/genomic_benchmark_mamba \
        dataset.dataset_name=demo_human_or_worm \
        wandb.id=demo_human_or_worm_hyenadna \
        train.pretrained_model_path=path/to/pretrained_model \
        trainer.devices=1
```

## Model Interpretation

GenBench now includes a model interpretation module that uses Captum to compute attributions of nucleotides to model predictions. This helps analyze which parts of a DNA sequence are most important for a model's decision.

### Using the Interpretation Module

You can run the interpretation module with:
```shell
python interpret.py experiment=hg38/genomic_benchmark_hyena \
        dataset.dataset_name=demo_human_or_worm \
        train.pretrained_model_path=path/to/pretrained_model \
        interpret.attribution_method=ig \
        interpret.num_samples=10 \
        interpret.motif_id=MA0108.1
```

### Interpretation Options

- `interpret.attribution_method`: Method to compute attributions ('saliency', 'ig' for Integrated Gradients, 'dl' for DeepLift)
- `interpret.num_samples`: Number of samples to analyze
- `interpret.motif_id`: JASPAR motif ID to correlate attributions with (optional)
- `interpret.target_class`: Target class for attribution (default: 0)
- `interpret.output_dir`: Directory to save interpretation results

### Visualization

The module generates visualizations that show:
- Attribution scores for each nucleotide position
- Correlation with known motifs from JASPAR database (if specified)

## Repeat the experiment
Please see [experiment.MD](experiment/experiment.MD) for the details of experiment management. and find scrips in 'experiment' directory


## Overview of Model Zoo and Datasets

We support various Genomic foundation models. We are working on add new methods and collecting experiment results.

* Spatiotemporal Prediction Methods.

    <details open>
    <summary>Currently supported methods</summary>

    - [x] [HyenaDNA](https://arxiv.org/abs/2306.15794) (NeurIPS'2023)
    - [x] [Caduceus](https://arxiv.org/abs/2403.03234) (Arxiv'2024)
    - [x] [DNABERT](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680) (Bioinformatics'2021)
    - [x] [DNABERT-2](https://arxiv.org/pdf/2306.15006.pdf) (Arxiv'2023)
    - [x] [The Nucleotide Transformer](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v3.abstract) (BioRxiv'2023)
    - [x] [Gena-LM](https://www.biorxiv.org/content/10.1101/2023.06.12.544594v1) (BioRxiv'2023)
   

    

* Genomic foundation models Benchmarks.

    <details open>
    <summary>Currently supported datasets</summary>

    - [x] [Genomic benchmark](https://www.biorxiv.org/content/10.1101/2022.06.08.495248v1.full) (BMC Genomic Data'2023) [[download](https://huggingface.co/datasets/katielink/genomic-benchmarks/tree/main)] [[config](experiment/genomic_benchmark)]
    - [x] [GUE](https://arxiv.org/pdf/2306.15006.pdf) (Arxiv'2023) [[download](https://drive.google.com/file/d/1GRtbzTe3UXYF1oW27ASNhYX3SZ16D7N2/view)] [[config](experiment/GUE)]
    - [x] [Promoter prediction](https://basespace.illumina.com/projects/66029966/about) (BioRxiv'2023) [[download](https://github.com/AIRI-Institute/GENA_LM/tree/main/downstream_tasks/promoter_prediction)] [[config](experiment/Promoter_prediction)]
    - [x] [Splice site prediction](https://dl.acm.org/doi/10.1177/0278364913491297) (Cell Press'2019) [[download](https://drive.google.com/file/d/1oB0DUrVfz-l0-wAe5_1kDVY5xZ1W4zMF/view?usp=share_link)] [[config](experiment/Splicing_prediction)]
    - [x] [Drosophila enhancer activity prediction](https://www.nature.com/articles/s41588-022-01048-5) (Nature Genetics'2022) [[download](https://data.starklab.org/almeida/DeepSTARR/Data/)] [[config](experiment/drosophila_enhancer_activity)]
    - [x] [Genomic Structure Prediction](https://www.nature.com/articles/s41588-022-01065-4) (Nature Genetics'2022) [[download](https://zenodo.org/record/6234936/files/resources_mcools.tar.gz)] [[config](experiment/genomic_structure)]
    

    </details>

<p align="right">(<a href="#top">back to top</a>)</p>

## Visualization

We present visualization examples of HyenaDNA below. For more detailed information, please refer to the notebook.

- For species classification task, visualization of t-sne embedding  can be found in [notebook/gene_cluster.ipynb](notebook/gene_cluster.ipynb). 


<!-- | :---: | | -->
<div align=center><img src='assets/dna_clustering.png' height="auto" width="260" ></div>


- For visualization of Bulk RNA Expression, please refer to [notebook/Bulk_prediction_spearman.ipynb](notebook/Bulk_prediction_spearman.ipynb).


 <div align=center><img src='notebook/Bulk_RNA_Expression.png' height="auto" width="260" ></div> 

- For Genomic Structure Prediction, visualization of predicted structures and ground truth structures are shown in [notebook/plot_genomic_structure_h1esc.ipynb](notebook/plot_genomic_structure_h1esc.ipynb) and [notebook/plot_genomic_structure_hff.ipynb](notebook/plot_genomic_structure_hff.ipynb) after running the experiment.

<div align=center>
<img src="assets/genomic_structure_h1esc.png" height="auto" width="260" >
</div>

<div align=center>
<img src="assets/genomic_structure_hff.png" height="auto" width="260" >
</div>




 


- for Drosophila enhancer activity prediction, visualization of predicted enhancers and ground truth enhancers are shown in [notebook/drosophila_pearsonr.ipynb](notebook/drosophila_pearsonr.ipynb) after running the experiment.

 <div align=center><img src="notebook/Drosophila_Enhancers_Prediction.png" height="auto" width="260" ></div> 

- for analysis of space complexity, please refer to [notebook/count_flops.ipynb](notebook/count_flops.ipynb) and for analysis of length effects and size effects, please refer to [notebook/performance_length.ipynb](notebook/performance_length.ipynb) and [notebook/parameter_size.ipynb](notebook/parameter_size.ipynb) respectively.




## License

This project is released under the [Apache 2.0 license](LICENSE). See `LICENSE` for more information.

## Acknowledgement

The framework of GenBench is insipred by [HyenaDNA](https://github.com/HazyResearch/hyena-dna)

## Contact 

- Jiahui Li(jiahuili.jimmy@gmail.com), Westlake University
- Zicheng Liu(liuzicheng@westlake.edu.cn), Westlake University
- Lei Xin(9201310419@stu.njau.edu.cn),Westlake University





<p align="right">(<a href="#top">back to top</a>)</p>
