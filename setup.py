from setuptools import setup, find_packages

setup(
    name="stats-cvm",
    version="0.1",
    packages=find_packages(),
    install_requires = [
        "pandas",
        "numpy",
        "matplotlib",
        "scipy"
    ],
    author="Andrew Lavigne",
    description="Une libraire sur mesure pour les cours de statistiques au Cégep du Vieux Montréal",
)