from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tacd-pisa1200",
    version="1.0.0",
    author="TACD Research Team",
    author_email="research@tacd-pisa1200.org",
    description="Mitigating Cross-Modal Hallucination in Vision-Language Models via Contrastive Decoding and Structural Topology Priors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tacd-pisa1200/tacd-pisa1200",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "pillow>=9.0.0",
        "numpy>=1.22.0",
    ],
    extras_require={
        "full": [
            "transformers>=4.35.0",
            "scipy>=1.9.0",
            "rich>=12.0.0",
            "tqdm>=4.64.0",
            "pytest>=7.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "tacd-cli=main:main",
            "pisa1200-eval=benchmarks.pisa1200_eval:main",
        ],
    },
)
