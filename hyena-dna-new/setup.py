from setuptools import setup, find_packages

setup(
    name="hyena-dna",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "transformers>=4.18.0",
        "einops>=0.4.1",
    ],
    author="GenBench Team",
    description="HyenaDNA: Эффективные модели для анализа ДНК",
    python_requires=">=3.7",
) 