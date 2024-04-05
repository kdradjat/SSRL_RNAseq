import pandas as pd

__all__ = ["load_tcga"]


def load_tcga(clinical_path, gene_expression_path):
    clinical = pd.read_parquet(clinical_path)
    gene_expression = pd.read_parquet(gene_expression_path)
    
    clinical.set_index("sampleID", inplace=True)
    gene_expression.set_index("caseID", inplace=True)
    
    gene_expression.index = gene_expression.index.str.split("-").str[:4].str.join("-")

    if not clinical.index.is_unique:
        raise ValueError

    if not gene_expression.index.is_unique:
        raise ValueError

    common_case = clinical.index.intersection(gene_expression.index)

    clinical = clinical.loc[common_case]
    gene_expression = gene_expression.loc[common_case]

    data = pd.concat({"clinical": clinical, "gene_expression": gene_expression}, axis=1)
    data.index.name = "caseID"

    return data
