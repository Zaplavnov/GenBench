from setuptools import setup, find_packages

setup(
    name="hyena_dna",
    version="0.1.0",
    packages=find_packages("hyena-dna"),
    package_dir={"": "hyena-dna"},
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "einops>=0.4.1",
        "transformers>=4.21.1",
    ],
) 