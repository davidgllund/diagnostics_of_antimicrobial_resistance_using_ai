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
- Polars >= 1.0.0
- NumPy >= 2.3.3
- Scikit-learn >= 1.3.1
- PyYAML >= 6.0
- tqdm >= 4.66

## Repository structure
- pyproject.toml: project metadata and editable-install configuration for a proper Python package.
- src/dire/: installable package entry points; thin compatibility wrappers that re-export the implementation from scripts/ so it can be imported as `dire.<module>` without duplicating the source.
- src/scripts/: packaged counterpart of scripts/, dynamically loading each module from the repository's scripts/ directory at import time (see src/scripts/_bridge.py) so the installed package and the working copy stay in sync.
- scripts/: implementation of the training, evaluation, fine-tuning, and analysis pipeline (model architecture, preprocessing, and the standalone scripts described below).
- model/: an example model configuration (config.yaml) plus the checkpoints and artifacts (trained model weights, vocabulary, calibration thresholds, training state, and evaluation/performance tables) produced by running the pipeline with it.
- model_finetune/: an example configuration for fine-tuning or lab-specific training on top of a base model.
- data/: contains the input data and preprocessing artifacts required by the pipeline.
- auxiliary_files/: reference lookup tables (antibiotic and species abbreviations, sample category labels) used during preprocessing and evaluation.
- notebooks/: exploratory analysis notebooks.

## Python project usage
Install the project in editable mode from the repository root:

```bash
python -m pip install -e .
```

Once installed, the project can be used through the package CLI or direct module execution:

```bash
python -m dire train --config model/config.yaml --folder ./experiments/example
python -m dire evaluate --config model/config.yaml --folder ./experiments/example
python -m dire fine-tune --config config.yaml --folder ./model_finetune
python -m dire process-data --input data/raw.tsv --output data/formatted.tsv --vocab data/vocab.pth
```

The following console commands are also available:

```bash
dire-train --config model/config.yaml --folder ./experiments/example
dire-evaluate --config model/config.yaml --folder ./experiments/example
dire-process-data --input data/raw.tsv --output data/formatted.tsv --vocab data/vocab.pth
```

## Additional analysis scripts
A few more specialized scripts live in scripts/ and are run directly rather than through the `dire` CLI:

```bash
python scripts/train_lab_specific_model.py --config config.yaml --folder ./model_finetune --lab SE100
python scripts/predict_new_sample.py --input isolates.txt --config config.yaml --folder ./experiments/example
python scripts/evaluate_standard_panels.py --input data/raw.tsv --config config.yaml --folder ./experiments/example --data data/formatted.tsv
python scripts/compute_resistance_correlations.py --input data/svebar/svebar_formatted.tsv --output data/svebar/resistance_correlations.tsv
```

- train_lab_specific_model.py: trains a lab-specific variant of the model (e.g. for lab SE100, SE300, or SE340), regularizing predictions toward per-lab class/group means.
- predict_new_sample.py: predicts resistance phenotypes for new, unlabeled isolates from raw isolate sentences.
- evaluate_standard_panels.py: evaluates model predictions restricted to the standard antibiotic testing panels actually used per lab/pathogen/specimen combination.
- compute_resistance_correlations.py: computes pairwise Pearson correlations of resistance status between antibiotics, optionally restricted to a species and/or specimen type.

## Workflow
A typical analysis pipeline is as follows:
1. Prepare or adapt the input dataset and vocabulary files referenced by the configuration.
2. Configure the model and training settings in the YAML file.
3. Train the model with the training script and save outputs to a dedicated experiment folder.
4. Run evaluation on validation or calibration data to produce phenotype predictions and probability outputs.
5. Optionally fine-tune the trained model, or train a lab-specific variant, on data from additional hospitals/labs.
6. Use the resulting predictions for downstream analysis (e.g. standard panel evaluation, resistance correlations), interpretation, or publication-ready figures.

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

