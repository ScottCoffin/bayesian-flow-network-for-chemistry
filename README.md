# ChemBFN: Bayesian Flow Network for Chemistry

[![arxiv](https://img.shields.io/badge/arXiv-2407.20294-red)](https://arxiv.org/abs/2407.20294)
[![arxiv](https://img.shields.io/badge/arXiv-2412.11439-red)](https://arxiv.org/abs/2412.11439)

This is the repository of the PyTorch implementation of ChemBFN model.

## Features

ChemBFN provides the state-of-the-art functionalities of
* SMILES or SELFIES-based *de novo* molecule generation
* Protein sequence *de novo* generation
* Classifier-free guidance conditional generation (single or multi-objective optimisation)
* Context-guided conditional generation (inpaint)
* Outstanding out-of-distribution chemical space sampling
* Molecular property and activity prediction finetuning
* Reaction yield prediction finetuning

in an all-in-one-model style.

## News

* [17/12/2024] The second paper of out-of-distribution generation is available on [arxiv.org](https://arxiv.org/abs/2412.11439).
* [31/07/2024] Paper is available on [arxiv.org](https://arxiv.org/abs/2407.20294).
* [21/07/2024] Paper was submitted to arXiv.

## Usage

You can find example scripts in [📁example](./example) folder.

## Pre-trained Model

You can find pretrained models in [release](https://github.com/Augus1999/bayesian-flow-network-for-chemistry/releases).

## Dataset Format

We provide a Python class [`CSVData`](./bayesianflow_for_chem/data.py) to handle data stored in CSV or similar format containing headers with the following tags:
* __smiles__ or __safe__ or __selfies__ or __geo2seq__ (_mandatory_): the entities under this tag should be molecule SMILES, SAFE, SELFIES or Geo2Seq strings. Multiple tags are acceptable (however, if "safe" or "geo2seq" is used, only the items under the last tag will be loaded).
* __value__ (_optional_): entities under this tag should be molecular properties or classes. Multiple tags are acceptable and in this case you can tell `CSVData` which value(s) should be loaded by specifying `label_idx=[...]`. If a property is not defined, leave it empty and the entity will be automatically masked to torch.inf telling the model that this property is unknown.

## Cite This Work

```bibtex
@misc{2024chembfn,
      title={A Bayesian Flow Network Framework for Chemistry Tasks}, 
      author={Nianze Tao and Minori Abe},
      year={2024},
      eprint={2407.20294},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.20294}, 
}
```
Out-of-distribution generation:
```bibtex
@misc{2024chembfn_ood,
      title={Bayesian Flow Is All You Need to Sample Out-of-Distribution Chemical Spaces}, 
      author={Nianze Tao},
      year={2024},
      eprint={2412.11439},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.11439}, 
}
```
