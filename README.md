<div align="center">

# Converting Your Thoughts to Texts: Enabling Brain Typing via Deep Feature Learning of EEG Signals

[![PerCom 2018](https://img.shields.io/badge/PerCom-2018-blue.svg)](https://ieeexplore.ieee.org/abstract/document/8444575)
[![arXiv](https://img.shields.io/badge/arXiv-1709.08820-b31b1b.svg)](https://arxiv.org/abs/1709.08820)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE)

**Xiang Zhang<sup>1</sup>, Lina Yao<sup>2</sup>, Quan Z. Sheng<sup>2</sup>, Salil S. Kanhere<sup>2</sup>, Tao Gu<sup>3</sup>, Dalin Zhang<sup>2</sup>**

*IEEE International Conference on Pervasive Computing and Communications (PerCom), 2018*

</div>

---

## Overview

Brain-computer interface (BCI) typing translates motor imagery EEG signals directly into
text commands without any physical movement. This work proposes a unified deep learning
framework that jointly trains a convolutional neural network (CNN) for spatial feature
extraction, a recurrent neural network (RNN) for temporal dependency modeling, and an
autoencoder (AE) for robustness against incomplete or corrupted EEG signals. The fused
CNN+RNN representations are passed to an XGBoost classifier for final class prediction.
The framework is validated on the PhysioNet EEG Motor Movement/Imagery Database and a
locally collected Emotiv dataset, and a working brain typing prototype is demonstrated
in a [video](https://www.youtube.com/watch?v=Dc0StUPq61k).

## Method

```
Raw EEG signals
      |
      v
+------------+
|    CNN     |  -- Conv + MaxPool + FC layers --> spatial features
+------------+
      |
      v
+------------+
|    RNN     |  -- Multi-layer LSTM --> temporal features
+------------+
      |
      +------- concatenate CNN + RNN features -------+
                                                      |
                                                      v
                                               +------------+
                                               | Autoencoder|  -- encoder -> compressed representation
                                               +------------+
                                                      |
                                                      v
                                               +------------+
                                               |  XGBoost   |  -- multi-class classification
                                               +------------+
                                                      |
                                                      v
                                               Predicted class
```

- **CNN block:** Two convolutional layers with ReLU activations and 1x2 max-pooling,
  followed by two fully connected layers with sigmoid activation and dropout.
- **RNN block:** Four hidden linear projection layers feed into a two-layer stacked LSTM
  with L2 regularization and Adam optimization.
- **Autoencoder:** A two-layer encoder-decoder trained with RMSProp to compress the
  concatenated CNN+RNN feature vector, improving robustness to signal noise.
- **XGBoost classifier:** Trained on the AE-encoded features using softmax multi-class
  objective with subsampling and depth-controlled trees.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

| Package       | Role                                      |
|---------------|-------------------------------------------|
| tensorflow    | CNN, RNN, and autoencoder training        |
| scipy         | Loading `.mat` data files                 |
| numpy         | Array operations and data preprocessing   |
| scikit-learn  | Z-score normalization (`preprocessing`)   |
| xgboost       | Final multi-class classifier              |

> **Note:** The code was written for Python 2 / TensorFlow 1.x. Running it under
> Python 3 / TensorFlow 2.x requires minor syntax updates (e.g., `print` statements,
> integer division).

## Data

Two EEG datasets are used. See [data/README.md](data/README.md) for download
instructions, file formats, label definitions, and path configuration.

- **PhysioNet eegmmidb** — large-scale public motor imagery dataset; `S1_nolabel6.mat`
  (one subject) is included as an example.
- **Emotiv lab dataset** — 7-subject, 5-class dataset collected with an Emotiv EPOC
  headset; `emotiv_7sub_5class.mat` is included.

## Usage

1. Place your `.mat` data file in the `data/` directory (or update the path in
   `src/model.py`; see `data/README.md`).

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the full pipeline:
   ```bash
   python main.py
   ```

   Training progress and per-step accuracy are printed to stdout. The pipeline runs
   CNN training, then RNN training, then autoencoder training, then XGBoost fitting
   in sequence.

## Repository Structure

```
Brain_typing/
├── main.py                  # Entry point — runs the full pipeline
├── requirements.txt         # Python dependencies
├── CITATION.cff             # Machine-readable citation metadata
├── LICENSE                  # CC BY-NC-SA 4.0
├── .gitignore
├── src/
│   ├── __init__.py
│   └── model.py             # CNN + RNN + AE + XGBoost pipeline
└── data/
    ├── README.md            # Dataset documentation and path configuration
    ├── S1_nolabel6.mat      # PhysioNet eegmmidb, Subject 1 example
    └── emotiv_7sub_5class.mat  # Emotiv lab dataset (7 subjects, 5 classes)
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhang2018converting,
  title     = {Converting Your Thoughts to Texts: Enabling Brain Typing via Deep Feature Learning of EEG Signals},
  author    = {Zhang, Xiang and Yao, Lina and Sheng, Quan Z. and Kanhere, Salil S. and Gu, Tao and Zhang, Dalin},
  booktitle = {2018 IEEE International Conference on Pervasive Computing and Communications (PerCom)},
  pages     = {1--10},
  year      = {2018},
  publisher = {IEEE},
  doi       = {10.1109/PERCOM.2018.8444575}
}
```

## Contact

For questions about the code or algorithm, contact **xiang.alan.zhang@gmail.com**.

## License

This repository is licensed under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](LICENSE).
