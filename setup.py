"""NatyaVeda Analyzer — Setup"""
from setuptools import setup, find_packages

setup(
    name="natyaveda-analyzer",
    version="1.0.0",
    description="AI-powered Indian Classical Dance Recognition & Analysis System",
    author="NatyaVeda Contributors",
    license="Apache-2.0",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Core ML
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "torchaudio>=2.2.0",
        # HuggingFace ecosystem
        "transformers>=4.40.0",
        "huggingface-hub>=0.22.0",
        "timm>=1.0.0",
        # TensorFlow (for MoveNet via TF Hub)
        "tensorflow>=2.15.0",
        "tensorflow-hub>=0.16.0",
        # MediaPipe (hand landmarks)
        "mediapipe>=0.10.0",
        # Computer Vision
        "opencv-python>=4.9.0",
        "Pillow>=10.0.0",
        # Pose estimation (MMPose ecosystem)
        "mmengine>=0.10.0",
        "mmcv>=2.1.0",
        "mmdet>=3.3.0",
        "mmpose>=1.3.0",
        # Video
        "yt-dlp>=2024.1.0",
        "scenedetect[opencv]>=0.6.3",
        # Data
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        # Training
        "scikit-learn>=1.3.0",
        "einops>=0.7.0",
        # Logging & config
        "wandb>=0.16.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "rich>=13.0.0",
        # Visualization
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.5.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "ipython>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "natyaveda-download=src.data_collection.youtube_downloader:main",
            "natyaveda-train=scripts.train:main",
            "natyaveda-eval=scripts.evaluate:main",
            "natyaveda-infer=scripts.infer:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video :: Analysis",
    ],
)
