from setuptools import find_packages, setup

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
    install_requires=["torch>=1.12", 
                      "tqdm>=4.64", 
                      'accelerate', 
                      'beartype',
                      'torchvision',
                      'numpy>=1.18.2',
                      'pandas>=1.0.3',
                      'keras>=2.3.1',
                      'argparse>=1.1',
                      'sklearn>=0.21.3',
                      'xgboost',
                      'tensorflow>=1.15.0'
                      ]
)
