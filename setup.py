from setuptools import setup

setup(
    name="SSRL_RNAseq",
    description="Self-supervised methods on gene expression data",
    author="Kevin Dradjat",
    url="https://github.com/kdradjat/SSRL_RNAseq",
    keywords=[
        "artificial intelligence",
        "contrastive learning",
        "self-supervised learning",
    ],
    python_requires=">=3.7",
    package_dir={"": "src"},
    packages=["ssrl_rnaseq"],
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "ipykernel",
    ]
)
