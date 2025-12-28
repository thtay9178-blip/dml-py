"""
DML-PY - A Collaborative Deep Learning Library
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dml-py",
    version="1.0.0",
    author="DML-PY Contributors",
    author_email="dml-py@example.com",
    description="A production-ready library for Deep Mutual Learning and collaborative neural network training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/DML-PY",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/DML-PY/issues",
        "Documentation": "https://github.com/yourusername/DML-PY/blob/main/README.md",
        "Source Code": "https://github.com/yourusername/DML-PY",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "validation_tests"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords="deep-learning mutual-learning knowledge-distillation pytorch collaborative-learning neural-networks",
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.13.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
            "sphinx>=6.2.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "optuna": [
            "optuna>=3.0.0",
        ],
        "onnx": [
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
        ],
        "all": [
            "optuna>=3.0.0",
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
            "jupyter>=1.0.0",
            "seaborn>=0.12.0",
        ],
    },
    include_package_data=True,
    package_data={
        "pydml": ["py.typed"],
    },
)
