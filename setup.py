from setuptools import setup

setup(
    name="dlimp",
    version="0.0.1",
    packages=["dlimp"],
    install_requires=[
        "tensorflow>=2.13.0", "tqdm", "tqdm-multiprocess==0.0.11"
    ],
)
