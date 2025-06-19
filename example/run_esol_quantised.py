# -*- coding: utf-8 -*-
"""Evaluate quantised ChemBFN on ESOL dataset.

The script assumes that the dataset `delaney-processed.csv` is available in
`../data/` and that the model checkpoint (`ChemBFN`) and a readout network
(`MLP`) are stored locally. The ChemBFN model is dynamically quantised before
inference to reduce memory usage.

Example:
    python run_esol_quantised.py --model_ckpt ../models/zinc15_190m.pt \
        --mlp_ckpt ../models/esol_mlp.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score

from bayesianflow_for_chem import ChemBFN, MLP
from bayesianflow_for_chem.data import CSVData, smiles2token, collate
from bayesianflow_for_chem.tool import split_dataset, quantise_model


THRESHOLD = -2.86  # median solubility used for binary classification


def load_esol(data_dir: Path) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare ESOL dataset.

    Parameters
    ----------
    data_dir: Path
        Directory containing ``delaney-processed.csv``.

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        Train, validation and test dataloaders.
    """
    dataset_file = data_dir / "delaney-processed.csv"
    if not (data_dir / "delaney-processed_train.csv").exists():
        split_dataset(dataset_file, method="scaffold")

    def encode(x):
        smiles = x["smiles"][0]
        value = float(x["measured log solubility in mols per litre"][0])
        label = 1 if value >= THRESHOLD else 0
        return {"token": smiles2token(smiles), "value": torch.tensor([label])}

    loaders = []
    for split in ["train", "val", "test"]:
        ds = CSVData(dataset_file.with_name(f"delaney-processed_{split}.csv"))
        ds.map(encode)
        loaders.append(DataLoader(ds, batch_size=32, collate_fn=collate))
    return tuple(loaders)  # type: ignore


def evaluate(model: ChemBFN, mlp: MLP, data: DataLoader) -> Tuple[float, float]:
    """Compute accuracy and ROC-AUC."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    mlp.to(device).eval()
    preds, labels = [], []
    with torch.inference_mode():
        for batch in data:
            x = batch["token"].to(device)
            y = batch["value"].to(device)
            logits = model.inference(x, mlp)
            prob = torch.softmax(logits, -1)[:, 1]
            preds.append(prob.cpu())
            labels.append(y.cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc = accuracy_score(labels, preds >= 0.5)
    auc = roc_auc_score(labels, preds)
    return acc, auc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=Path, required=True, help="ChemBFN checkpoint")
    parser.add_argument("--mlp_ckpt", type=Path, required=True, help="MLP checkpoint")
    parser.add_argument("--data_dir", type=Path, default=Path("../data"), help="dataset directory")
    args = parser.parse_args()

    train_loader, val_loader, test_loader = load_esol(args.data_dir)

    model = ChemBFN.from_checkpoint(args.model_ckpt)
    mlp = MLP.from_checkpoint(args.mlp_ckpt)

    model_q = quantise_model(model)

    acc, auc = evaluate(model_q, mlp, test_loader)
    print(f"Test accuracy: {acc:.4f}\nROC-AUC: {auc:.4f}")


if __name__ == "__main__":
    main()

