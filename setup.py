from setuptools import find_packages, setup  # type: ignore

__version__ = "1.0"

setup(
    name="neural_network",
    version=__version__,
    description="A neural network library",
    url="https://github.com/javidahmed64592/neural-network",
    author="Javid Ahmed",
    author_email="javidahmed@icloud.com",
    packages=find_packages(),
    install_requires=["numpy"],
)
