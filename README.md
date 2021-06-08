## Transfer-Based Semantic Anomaly Detection

This repository contains PyTorch code for two transfer-based anomaly detection (AD) models:

- AD with an inductive bias (ADIB),
- AD with residual adaptation (ADRA).

Change the configuration in `util/parser.py` or by launching `train.py` with custom commands, e.g. to train ADRA on class seven (horse) of the CIFAR-10 one-versus-rest AD benchmark:

`python3 train.py --model adra --normal_class 7`

ADIB uses L2SP to regularize model weights and is active by default. For experiments on the CIFAR-10 semantic AD benchmark, set `--benchmark hold_one_out`.

Data will initially be downloaded to `data` unless some path is provided via `config.data_path` that contains both CIFAR-10 and CIFAR-100. Both models require pretrained weights, which will download automatically provided the prerequisite `gdown` has been installed.

If you find that this code is useful in your research, please cite our work as:

```
@inproceedings{deecke21,
    author       = "Deecke, Lucas and Ruff, Lukas and Vandermeulen, Robert A. and Bilen, Hakan",
    title        = "Transfer-Based Semantic Anomaly Detection",
    booktitle    = "International Conference on Machine Learning",
    year         = "2021"
}
```
