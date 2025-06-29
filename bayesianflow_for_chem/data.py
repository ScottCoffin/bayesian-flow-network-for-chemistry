# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
Tokenise SMILES/SAFE/SELFIES/GEO2SEQ/protein-sequence strings.
"""
import os
import re
from pathlib import Path
from typing import Any, List, Dict, Union, Callable
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

__filedir__ = Path(__file__).parent

SMI_REGEX_PATTERN = (
    r"(\[|\]|H[e,f,g,s,o]?|"
    r"L[i,v,a,r,u]|"
    r"B[e,r,a,i,h,k]?|"
    r"C[l,a,r,o,u,d,s,n,e,m,f]?|"
    r"N[e,a,i,b,h,d,o,p]?|"
    r"O[s,g]?|S[i,c,e,r,n,m,b,g]?|"
    r"K[r]?|T[i,c,e,a,l,b,h,m,s]|"
    r"G[a,e,d]|R[b,u,h,e,n,a,f,g]|"
    r"Yb?|Z[n,r]|P[t,o,d,r,a,u,b,m]?|"
    r"F[e,r,l,m]?|M[g,n,o,t,c,d]|"
    r"A[l,r,s,g,u,t,c,m]|I[n,r]?|"
    r"W|X[e]|E[u,r,s]|U|D[b,s,y]|"
    r"b|c|n|o|s|p|"
    r"\(|\)|\.|=|#|-|\+|\\|\/|:|"
    r"~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
)
SEL_REGEX_PATTERN = r"(\[[^\]]+]|\.)"
GEO_REGEX_PATTERN = (
    r"(H[e,f,g,s,o]?|"
    r"L[i,v,a,r,u]|"
    r"B[e,r,a,i,h,k]?|"
    r"C[l,a,r,o,u,d,s,n,e,m,f]?|"
    r"N[e,a,i,b,h,d,o,p]?|"
    r"O[s,g]?|S[i,c,e,r,n,m,b,g]?|"
    r"K[r]?|T[i,c,e,a,l,b,h,m,s]|"
    r"G[a,e,d]|R[b,u,h,e,n,a,f,g]|"
    r"Yb?|Z[n,r]|P[t,o,d,r,a,u,b,m]?|"
    r"F[e,r,l,m]?|M[g,n,o,t,c,d]|"
    r"A[l,r,s,g,u,t,c,m]|I[n,r]?|"
    r"W|X[e]|E[u,r,s]|U|D[b,s,y]|"
    r"-|.| |[0-9])"
)
AA_REGEX_PATTERN = r"(A|B|C|D|E|F|G|H|I|K|L|M|N|P|Q|R|S|T|V|W|Y|Z|-|.)"
smi_regex = re.compile(SMI_REGEX_PATTERN)
sel_regex = re.compile(SEL_REGEX_PATTERN)
geo_regex = re.compile(GEO_REGEX_PATTERN)
aa_regex = re.compile(AA_REGEX_PATTERN)


def load_vocab(
    vocab_file: Union[str, Path]
) -> Dict[str, Union[int, List[str], Dict[str, int]]]:
    """
    Load vocabulary from source file.

    :param vocab_file: file that contains vocabulary
    :type vocab_file: str | pathlib.Path
    :return: {"vocab_keys": vocab_keys, "vocab_count": vocab_count, "vocab_dict": vocab_dict}
    :rtype: dict
    """
    with open(vocab_file, "r", encoding="utf-8") as f:
        lines = f.read().strip()
    vocab_keys = lines.split("\n")
    vocab_count = len(vocab_keys)
    vocab_dict = dict(zip(vocab_keys, range(vocab_count)))
    return {
        "vocab_keys": vocab_keys,
        "vocab_count": vocab_count,
        "vocab_dict": vocab_dict,
    }


_DEFUALT_VOCAB = load_vocab(__filedir__ / "vocab.txt")
VOCAB_KEYS: List[str] = _DEFUALT_VOCAB["vocab_keys"]
VOCAB_DICT: Dict[str, int] = _DEFUALT_VOCAB["vocab_dict"]
VOCAB_COUNT: int = _DEFUALT_VOCAB["vocab_count"]
AA_VOCAB_KEYS = (
    VOCAB_KEYS[0:3] + "A B C D E F G H I K L M N P Q R S T V W Y Z - .".split()
)
AA_VOCAB_COUNT = len(AA_VOCAB_KEYS)
AA_VOCAB_DICT = dict(zip(AA_VOCAB_KEYS, range(AA_VOCAB_COUNT)))
GEO_VOCAB_KEYS = VOCAB_KEYS[0:3] + [" "] + VOCAB_KEYS[22:150] + [".", "-"]
GEO_VOCAB_COUNT = len(GEO_VOCAB_KEYS)
GEO_VOCAB_DICT = dict(zip(GEO_VOCAB_KEYS, range(GEO_VOCAB_COUNT)))


def smiles2vec(smiles: str) -> List[int]:
    """
    SMILES tokenisation using a dataset-independent regex pattern.

    :param smiles: SMILES string
    :type smiles: str
    :return: tokens w/o `<start>` and `<end>`
    :rtype: list
    """
    tokens = [token for token in smi_regex.findall(smiles)]
    return [VOCAB_DICT[token] for token in tokens]


def geo2vec(geo2seq: str) -> List[int]:
    """
    Geo2Seq tokenisation using a dataset-independent regex pattern.

    :param geo2seq: Geo2Seq string
    :type geo2seq: str
    :return: tokens w/o `<start>` and `<end>`
    :rtype: list
    """
    tokens = [token for token in geo_regex.findall(geo2seq)]
    return [GEO_VOCAB_DICT[token] for token in tokens]


def aa2vec(aa_seq: str) -> List[int]:
    """
    Protein sequence tokenisation using a dataset-independent regex pattern.

    :param aa_seq: protein (amino acid) sequence
    :type aa_seq: str
    :return: tokens w/o `<start>` and `<end>`
    :rtype: list
    """
    tokens = [token for token in aa_regex.findall(aa_seq)]
    return [AA_VOCAB_DICT[token] for token in tokens]


def split_selfies(selfies: str) -> List[str]:
    """
    SELFIES tokenisation.

    :param selfies: SELFIES string
    :type selfies: str
    :return: SELFIES vocab
    :rtype: list
    """
    return [token for token in sel_regex.findall(selfies)]


def smiles2token(smiles: str) -> Tensor:
    # start token: <start> = 1; end token: <esc> = 2
    return torch.tensor([1] + smiles2vec(smiles) + [2], dtype=torch.long)


def geo2token(geo2seq: str) -> Tensor:
    # start token: <start> = 1; end token: <esc> = 2
    return torch.tensor([1] + geo2vec(geo2seq) + [2], dtype=torch.long)


def aa2token(aa_seq: str) -> Tensor:
    # start token: <start> = 1; end token: <end> = 2
    return torch.tensor([1] + aa2vec(aa_seq) + [2], dtype=torch.long)


def collate(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """
    Padding the data in one batch into the same size.\n
    Should be passed to `~torch.utils.data.DataLoader` as `DataLoader(collate_fn=collate, ...)`.

    :param batch: a list of data (one batch)
    :type batch: list
    :return: batched {"token": token} or {"token": token, "value": value}
    :rtype: dict
    """
    token = [i["token"] for i in batch]
    if "MAX_PADDING_LENGTH" in os.environ:
        lmax = int(os.environ["MAX_PADDING_LENGTH"])
    else:
        lmax = max([len(w) for w in token])
    token = torch.cat(
        [F.pad(i, (0, lmax - len(i)), value=0)[None, :] for i in token], 0
    )
    out_dict = {"token": token}
    if "value" in batch[0]:
        out_dict["value"] = torch.cat([i["value"][None, :] for i in batch], 0)
    if "mask" in batch[0]:
        mask = [i["mask"] for i in batch]
        out_dict["mask"] = torch.cat(
            [F.pad(i, (0, lmax - len(i)), value=0)[None, :] for i in mask], 0
        )
    return out_dict


class CSVData(Dataset):
    def __init__(self, file: Union[str, Path]):
        """
        Define dataset stored in CSV file.

        :param file: dataset file name <file>
        :type file: str | pathlib.Path
        """
        super().__init__()
        with open(file, "r") as db:
            self.data = db.readlines()
        self.header_idx_dict: Dict[str, List[int]] = {}
        for key, i in enumerate(self.data[0].replace("\n", "").split(",")):
            if i in self.header_idx_dict:
                self.header_idx_dict[i].append(key)
            else:
                self.header_idx_dict[i] = [key]
        self.mapping = lambda x: x

    def __len__(self) -> int:
        return len(self.data) - 1

    def __getitem__(self, idx: Union[int, Tensor]) -> Dict[str, Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # valid `idx` should start from 1 instead of 0
        data: List[str] = self.data[idx + 1].replace("\n", "").split(",")
        data_dict: Dict[str, List[str]] = {}
        for key in self.header_idx_dict:
            data_dict[key] = [data[i] for i in self.header_idx_dict[key]]
        return self.mapping(data_dict)

    def map(self, mapping: Callable[[Dict[str, List[str]]], Any]) -> None:
        """
        Pass a customised mapping function to transform the data entities to tensors.

        e.g.
        ```python
        import torch
        from bayesianflow_for_chem.data import smiles2token, CSVData


        def encode(x):
            return {
                "token": smiles2token(".".join(x["smiles"])),
                "value": torch.tensor([float(i) if i != "" else torch.inf for i in x["value"]]),
            }

        dataset = CSVData(...)
        dataset.map(encode)
        ```

        :param mapping: customised mapping function
        :type mapping: callable
        :return:
        :rtype: None
        """
        self.mapping = mapping


if __name__ == "__main__":
    ...
