"""NatyaVeda Analyzer v2 — Setup"""
from setuptools import setup, find_packages
setup(
    name="natyaveda-analyzer", version="2.0.0",
    description="Indian Classical Dance Recognition — RTMW-x+VitPose+MediaPipe+VideoMAE+DanceFormer",
    author="NatyaVeda Contributors", license="Apache-2.0",
    python_requires=">=3.10",
    packages=find_packages(where="src"), package_dir={"": "src"},
    install_requires=[
        "torch>=2.2.0","torchvision>=0.17.0","numpy>=1.24.0,<2.0.0",
        "transformers>=4.40.0","huggingface-hub>=0.22.0","timm>=1.0.0",
        "accelerate>=0.27.0","einops>=0.7.0","mediapipe>=0.10.0",
        "opencv-python-headless>=4.9.0","Pillow>=10.0.0",
        "yt-dlp>=2024.1.0","scenedetect>=0.6.3","ffmpeg-python>=0.2.0",
        "scikit-learn>=1.3.0","scipy>=1.11.0","pandas>=2.0.0",
        "pyyaml>=6.0","omegaconf>=2.3.0","rich>=13.0.0","tqdm>=4.65.0",
        "matplotlib>=3.7.0","seaborn>=0.12.0","wandb>=0.16.0","tensorboard>=2.15.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0","pytest-cov>=4.1.0","black>=23.0.0","isort>=5.12.0","flake8>=6.0.0"],
        "mmpose": ["mmengine>=0.10.0"],
        "tf": ["tensorflow-cpu>=2.15.0","tensorflow-hub>=0.16.0"],
        "notebook": ["jupyter>=1.0.0","ipywidgets>=8.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
    ],
)
