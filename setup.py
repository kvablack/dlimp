from setuptools import setup

setup(
    name="dlimp",
    version="0.0.1",
    packages=["dlimp"],
    install_requires=[
        "tensorflow==2.11.0", "tqdm==4.65.0", "tqdm-multiprocess==0.0.11"
    ],
)