# Unified Uncertainty Calibration
Download the paper [here](https://arxiv.org/abs/2310.01202).

## Usage
1. Set your data root directory in `datasets.ROOT_`
1. Download datasets using `python download.py`, and the instructions printed therein.
1. Precompute last-layer features using `python precompute.py`
1. Run all methods using `python main.py --sweep`
1. Print tables of results using `python table.py`

## Citation
```
@article{u2c,
  title={Unified Uncertainty Calibration},
  author={Chaudhuri, Kamalika and Lopez-Paz, David},
  year={2023},
  journal={arXiv}
}
```

## License

Work released under the Apache License 2.0. See [LICENSE](LICENSE) for additional details.
