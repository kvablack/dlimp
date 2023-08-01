from setuptools import setup

setup(
    name="dlimp",
    version="0.0.1",
    packages=["dlimp"],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.13.0",
        "tqdm",
        "tqdm-multiprocess==0.0.11",
        "pre-commit==3.3.3",
    ],
)
