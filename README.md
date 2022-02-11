# LAMANAS: Loss Agnostic and Model Agnostic Meta Neural Architecture Search for Few-Shot Learning


This is an implmentation of [our paper](https://cs229.stanford.edu/proj2021spr/report2/82285254.pdf) on Loss Agnostic and Model Agnostic Meta Neural Architecture Search (LAMANAS) using a self-supervised loss, parameterized by a neural network. Our base implemention is derived from [Meta-Learning of Neural Architectures for Few-Shot Learning](https://openaccess.thecvf.com/content_CVPR_2020/html/Elsken_Meta-Learning_of_Neural_Architectures_for_Few-Shot_Learning_CVPR_2020_paper.html), located [here](https://github.com/boschresearch/metanas/tree/305e3070908c6adf974fbdf8220e8afba4eb60fd).



## Requirements and Setup

### Install requiered packages.
Run

```bash
conda env create -f environment.yml
```
to create a new conda environment named `metanas` with all requiered packages and activate it.

### Download the data

Download the data sets you want to use (Omniglot or miniImagenet). You can also set `download=True` for the data loaders in [`torchmeta_loader.py`](metanas/tasks/torchmeta_loader.py) to use the data download provided by [Torchmeta](https://github.com/tristandeleu/pytorch-meta). 



## How to Use

Please refer to the [`scripts`](scripts/) folder for examples how to use this code. E.g., for experiments on miniImagenet:

- Running meta training for MetaNAS: [`run_in_meta_train.sh`](scripts/run_in_meta_train.sh)
- Running meta testing for a checkpoint from the above meta training experiment: [`run_in_meta_testing.sh`](scripts/run_in_meta_testing.sh)
- Scaling up an optimized architecture from above meta training experiment and retraining it: [`run_in_upscaled.sh`](scripts/run_in_upscaled.sh)
