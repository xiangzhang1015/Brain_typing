# Data

This project uses two EEG datasets.

---

## 1. PhysioNet EEG Motor Movement/Imagery Database (eegmmidb)

**Description:** A large-scale public dataset of motor imagery EEG recordings. The file
`S1_nolabel6.mat` included in this repository is a single-subject excerpt (Subject 1)
from the PhysioNet eegmmidb collection. Each row contains 64 EEG channel features plus
a class label in column 65. The excerpt covers 6 motor imagery classes.

**Download:** Full dataset available at https://physionet.org/content/eegmmidb/1.0.0/

**Citation:**
> Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet: components of a
> new research resource for complex physiologic signals. Circulation. 2000;101(23):e215–e220.

**Placement:** Place `.mat` files in this `data/` directory.

---

## 2. Emotiv Lab Dataset

**Description:** A locally collected dataset recorded with an Emotiv EPOC headset
(14 channels, 128 Hz sampling rate). The file `emotiv_7sub_5class.mat` contains
recordings from 7 subjects across 5 action classes. Each row has 16 columns:
columns 1–14 are raw EEG channel values (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8,
FC6, F4, F8, AF4), column 15 is the action label, and column 16 is the subject label.

**Action labels:**
| Label | Action     |
|-------|------------|
| 1     | Up         |
| 2     | Down       |
| 3     | Left       |
| 4     | Right      |
| 5     | Middle (excluded in paper due to low SNR) |
| 6     | Eye close  |

The PerCom 2018 paper uses labels 1, 2, 3, 4, 6 (five classes), relabeled 0–4.

**Subjects:** 7 participants (Xiang Zhang, Xigang Yuan, Xiaoqing Zhang, Dalin Zhang,
Xiaodong Ning, Manqing Dong, Kaixuan Chen).

**Placement:** Place `emotiv_7sub_5class.mat` in this `data/` directory.

---

## Updating Data Paths

The data path is set near the top of `src/model.py`. Change the `loadmat` call to
point to your local file:

```python
# src/model.py, line ~21
feature = sc.loadmat("data/S1_nolabel6.mat")
```

Replace `"data/S1_nolabel6.mat"` with the absolute or relative path to your `.mat` file.
For the Emotiv dataset, load `data/emotiv_7sub_5class.mat` and adjust the feature/label
extraction indices accordingly (columns 0–13 for features, column 14 for labels).
