from setuptools import find_packages, setup

with open("app/README.md", "r") as f:
    long_description = f.read()

setup(
    name="mmm-pypack",
    version="0.0.1",
    description="Pacote Python com classes e funções recorrentes no MMM.",
    # package_dir={"": "app"},
    # packages=find_packages(where="app"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GabrielMDePaoli/mmm-pypack",
    author="Squad MMM",
    author_email="mousse@arjancodes.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=["pandas","numpy","scikit-learn","termcolor","joblib","pickle","statsmodels"],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.10",
)
