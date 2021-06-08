## Transfer-Based Semantic Anomaly Detection.

This repository contains code for two transfer-based anomaly detection (AD) models:

- AD with an inductive bias (ADIB),
- AD with residual adaptation (ADRA).

Change the configuration in `util/parser.py` or by launching `train.py` with custom commands, e.g. to train ADRA:

`python3 train.py --ra --normal_class 7`

Data will initially be downloaded to `data` unless some path is provided via `config.data_path` that contains both CIFAR-10 and CIFAR-100. Both models require pretrained weights, which will download automatically provided `gdown` has been installed.

If you find this code useful in your research, cite our work as

```
@inproceedings{deecke21,
    author       = "Deecke, Lucas and Ruff, Lukas and Vandermeulen, Robert A. and Bilen, Hakan",
    title        = "Transfer-Based Semantic Anomaly Detection",
    booktitle    = "International Conference on Machine Learning",
    year         = "2021"
}
```
