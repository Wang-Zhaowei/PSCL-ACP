# PSCL-ACP: Pretrained Sequence–structure Collaborative Learning for Identifying Anticancer Peptides and Their Cancer-type Activities

## PSCL-ACP

An Integrated Two Stages Framework for Anticancer Peptide Discovery and Activity Prediction.

## Overview

PSCL-ACP is a Two Stages computational pipeline for anticancer peptide (ACP) discovery:

Stage 1: Binary ACP classification

Stage 2: Multi-label cancer-type prediction

## Repository Structure

```text
/
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
└── datasets/      
```

## Requirement

```bash
pip install -r requirements.txt
```

## Datasets

### Binary

The ACP-Mixed-80 dataset is used for binary anticancer peptide (ACP) identification. The training set consists of 242 ACPs and 242 non-ACPs, and the independent test set consists of 61 ACPs and 61 non-ACPs.

-Training dataset: datasets/ACP-Mixed-80/ACP-Mixed-80-train.tsv
-independent testing dataset: datasets/ACP-Mixed-80/ACP-Mixed-80-test.tsv

### Multi-label Classification

The multi-label dataset is used to evaluate the prediction of cancer-type activities for ACPs. It contains 728 unique peptide sequences, each of which is annotated with one or more cancer types according to its reported anticancer activity. Seven cancer types are considered, including Breast, Cervix, Colon, Leukemia, Liver, Lung, and Prostate. We have already split the dataset into ten folds for cross-validation.

```text
datasets/ACP0819_10fold/train_1.tsv
datasets/ACP0819_10fold/test_1.tsv
...
datasets/ACP0819_10fold/train_10.tsv
datasets/ACP0819_10fold/test_10.tsv
```

## Binary ACP classification

To identify whether peptides are ACPs, run:

```bash
python binary.py SSCR
python binary.py SHTL
python binary.py CCI
```

## Multi-label cancer-type prediction

To predict whether ACPs show activity against the seven cancer types (Breast, Cervix, Colon, Leukemia, Liver, Lung, and Prostate), run:

```bash
python multilabel.py
```
## Note
This repository does not contain the peptide tertiary structure files predicted by ESMFold. You can use ESMFold to predict 3D structures based on the provided FASTA peptide sequence files and save the generated PDB files locally for further analysis.
