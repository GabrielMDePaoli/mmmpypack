from setuptools import find_packages, setup

with open("app/README.md", "r") as f:
    long_description = f.read()

setup(
    name="mmmpypack",
    version="0.0.2",
    description="Pacote Python com classes e funções recorrentes no MMM.",
    package_dir={"": "app"},
    packages=find_packages(where="app"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GabrielMDePaoli/mmmpypack",
    author="Squad MMM",
    author_email="mousse@claro.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=["pandas>=2.0.2","numpy>=1.24.3","scikit-learn>=1.2.2","termcolor>=2.3.0","joblib>=1.2.0","statsmodels>=0.14.0", "feature_engine>=1.6.1"],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.9",
)
