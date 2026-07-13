# AMR prediction with transformer models

This repository contains the implementation used to train and evaluate a transformer-based model for antimicrobial resistance (AMR) prediction from clinical text data. It is structured as a reproducible research codebase and is intended to accompany a scientific publication by providing the full training, evaluation, and analysis pipeline used in the study.

## Overview
The project focuses on predicting antimicrobial susceptibility phenotypes from structured clinical text representations. The implementation combines:
- a transformer-based language model for sequence encoding,
- antibiotic-specific classification heads for phenotype prediction,
- metadata-aware preprocessing and batching utilities,
- training and evaluation workflows for model development and performance assessment.

The code is organized around a configuration-driven workflow, where model hyperparameters, training settings, dataset locations, and vocabulary resources are defined in YAML configuration files.

## Repository purpose
This repository is designed to support three main objectives:
1. Reproducible training of AMR prediction models.
2. Systematic evaluation of predictive performance and uncertainty estimates.
3. Transparent sharing of the implementation details underlying the scientific work.

## Dependencies
The following software is required to run the pipeline:
- Python >= 3.11.3
- PyTorch >= 2.1.2
- Transformers >= 4.39.3
- Pandas >= 2.3.2
- NumPy >= 2.3.3
- Scikit-learn >= 1.3.1

## Repository structure
- scripts/aux.py: core helper functions for dataset construction, vocabulary handling, sample expansion, and batch preparation.
- scripts/evaluate_model.py: performs inference on evaluation or calibration splits, generates predictions and probability estimates, and supports conformal-style thresholding.
- scripts/model_utils.py: initializes the model architecture, optimizer and loss functions, and loads or saves training checkpoints and state.
- scripts/model.py: defines the transformer encoder and antibiotic-specific classification heads used by the prediction model.
- scripts/train_model.py: runs the training loop across epochs, computes training and validation metrics, and writes logs and model outputs.
- model/config.yaml: an example model configuration file used to define architecture and training behavior.
- data/: contains the input data and preprocessing artifacts required by the pipeline.

## Workflow
A typical analysis pipeline is as follows:
1. Prepare or adapt the input dataset and vocabulary files referenced by the configuration.
2. Configure the model and training settings in the YAML file.
3. Train the model with the training script and save outputs to a dedicated experiment folder.
4. Run evaluation on validation or calibration data to produce phenotype predictions and probability outputs.
5. Use the resulting predictions for downstream analysis, interpretation, or publication-ready figures.

## Key implementation details
The implementation includes several features that are relevant to the study workflow:
- transformer-based encoding of tokenized clinical text,
- multi-class or binary classification heads depending on the antibiotic-specific task,
- integration of non-text metadata during classification,
- balanced sampling and batch construction for training and evaluation,
- checkpointing and resumption of training from prior runs,
- probability-based evaluation and threshold calibration utilities.

## Outputs
The training and evaluation scripts generate model checkpoints, training logs, validation performance summaries, and prediction tables that can be used for further analysis and reporting.

## Reproducibility notes
To support reproducibility, the repository uses fixed random seeds for training and stores experiment artifacts in dedicated output folders. The configuration-driven design also makes it easier to rerun experiments and compare model variants.

## Notes for publication use
This repository is intended to be a faithful implementation of the methods described in the associated study. When using the code in a manuscript, supplementary material, or derivative work, it is recommended to refer to the accompanying publication and to preserve the configuration files and trained artifacts used in the reported experiments.

