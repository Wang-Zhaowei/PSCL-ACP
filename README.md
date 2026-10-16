# PSCL-ACP: Pretrained Sequence–structure Collaborative Learning for Identifying Anticancer Peptides and Their Cancer-type Activities

## Overview

PSCL-ACP is a pretrained sequence–structure collaborative learning framework for anticancer peptide (ACP) analysis. It supports two downstream tasks:

- **Binary ACP identification:** distinguishing ACPs from non-ACPs.
- **Multi-label cancer-type activity prediction:** predicting the cancer types against which an ACP may exhibit activity.

PSCL-ACP jointly models contextual sequence semantics and residue-level structural topology, and integrates the two modalities through cross-modal collaborative learning.

## Main Components

PSCL-ACP consists of three core modules:

- **SSCR (Semantics-aware Sequence Context Representation):** refines pretrained ESM2 residue embeddings with a Transformer encoder to capture ACP-related contextual sequence patterns.
- **SHTL (Structure-guided Hierarchical Topological Learning):** constructs residue-level graphs from ESMFold-predicted peptide structures and learns structural representations across multiple topological neighborhoods using progressive graph convolution and adaptive neighborhood fusion.
- **CCI (Complementarity-enhanced Cross-modal Integration):** promotes bidirectional interaction and complementary fusion between sequence and structural representations through fine- and coarse-grained cross-modal modeling.

The framework adopts a **two-stage training strategy**. SSCR and SHTL are first pretrained independently with task-specific objectives, after which the pretrained encoders are jointly optimized with CCI for downstream prediction.

## Repository Structure

```text
/PSCL-ACP
├── SSCR.py
├── SHTL.py
├── CCI.py
├── binary.py
├── multilabel.py
├── runner.py
├── data.py
├── metric.py
├── util.py
├── main.py
├── test.py
├── requirements.txt
└── datasets/
```

## Requirements

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

The experiments reported in the paper were conducted with Python 3.8 and PyTorch on Ubuntu 22.04 with CUDA 11.8.

## Datasets

### Binary ACP Identification

The ACP-Mixed-80 dataset is used for binary ACP identification. The training set contains 484 peptides, including 242 ACPs and 242 non-ACPs, and the independent test set contains 122 peptides, including 61 ACPs and 61 non-ACPs.

```text
datasets/ACP-Mixed-80/ACP-Mixed-80-train.tsv
datasets/ACP-Mixed-80/ACP-Mixed-80-test.tsv
```

### Multi-label Cancer-type Activity Prediction

The multi-label dataset contains 728 unique ACP sequences, each annotated with one or more cancer types according to its reported anticancer activity. Seven cancer-type labels are considered:

- Breast
- Cervix
- Colon-Rectum
- Leukemia
- Liver
- Lung
- Prostate

The dataset is organized into ten folds for cross-validation.

```text
datasets/ACP0819_10fold/train_1.tsv
datasets/ACP0819_10fold/test_1.tsv
...
datasets/ACP0819_10fold/train_10.tsv
datasets/ACP0819_10fold/test_10.tsv
```

## Running the Models

### Binary ACP Identification

Run the corresponding model configuration with:

```bash
python binary.py SSCR
python binary.py SHTL
python binary.py CCI
```

### Multi-label Cancer-type Activity Prediction

Run:

```bash
python multilabel.py
```

## Peptide Structure Preparation

PSCL-ACP uses predicted three-dimensional peptide structures as structural input to the SHTL module. In our experiments, these structures were generated locally using **ESMFold** and stored as PDB files.

Because the complete set of predicted structure files is too large to host conveniently in this GitHub repository, the PDB files used in the experiments are not included. They can be reproduced directly from the peptide sequences provided in the datasets.

For each peptide:

1. Convert the peptide sequence to FASTA format.
2. Predict its three-dimensional structure using ESMFold.
3. Save the predicted structure as an individual `.pdb` file.
4. Keep a consistent correspondence between the peptide identifier and the PDB filename so that the structure can be matched to the correct sequence during data loading.

The structures can be generated using either the official ESMFold implementation provided through `fair-esm` or the Hugging Face Transformers implementation of the `facebook/esmfold_v1` model.

The SHTL module subsequently constructs residue-level structural graphs from the predicted structures using the Cα coordinates of amino acid residues. Therefore, no experimentally determined PDB structures are required to reproduce the structural inputs used by PSCL-ACP.

