# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
Scorers.
"""
from rdkit import RDLogger
from rdkit.Contrib.SA_Score import sascorer  # type: ignore
from rdkit.Chem import MolFromSmiles, QED

RDLogger.DisableLog("rdApp.*")


def smiles_valid(smiles: str) -> int:
    """
    Return the validity of a SMILES string.

    :param smiles: SMIlES string
    :return: validity
    """
    return 1 if (MolFromSmiles(smiles) and smiles) else 0


def qed_score(smiles: str) -> float:
    """
    Return the quantitative estimate of drug-likeness score of a SMILES string.

    :param smiles: SMILES string
    :return: QED score
    """
    return QED.qed(MolFromSmiles(smiles))


def sa_score(smiles: str) -> float:
    """
    Return the synthetic accessibility score of a SMILES string.

    :param smiles: SMILES string
    :return: SA score
    """
    return sascorer.calculateScore(MolFromSmiles(smiles))


if __name__ == "__main__":
    ...
