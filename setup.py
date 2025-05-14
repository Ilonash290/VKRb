from setuptools import setup, find_packages

setup(
    name="salesman_optimization",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy==2.2.3',
        'networkx==3.4.2',
        'matplotlib==3.10.0',
        'scikit-learn==1.6.1',
        'seaborn==0.13.2'
    ],
    author="Ilona Shlyakhtina",
    description="A library for optimizing the number of salesmans",
    python_requires='>=3.13',
)
