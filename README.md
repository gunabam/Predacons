# Predacons
Natural Language Processing Repository. This package leverages huggingface transformer models with the ability to finetune on multiple tasks for curriculum learning. It leverages the InMemoryDataset from pytorch geometric for handling multi-input data.

## Installation
Install the package via pip symlinks.
```
conda env create -f environment.yml
conda activate nlp
pip install -e .
```

## Additional Notes
1. Example datasets can be found [here](https://github.com/magarveylab/PredaconDatasets2.0).
2. Example training scripts can be found [here](https://github.com/magarveylab/BaryonPredacons).