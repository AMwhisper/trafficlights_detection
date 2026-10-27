# Traffic Light State Recognition with Classical Machine Learning

EECE 5644 (Machine Learning and Pattern Recognition), Fall 2026 — Term Project
**Team:** Jingqi Mei, Andrew Chen

## Overview

This project classifies cropped traffic lights from the [LISA Traffic Light Dataset](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset) into six classes: `go`, `goLeft`, `stop`, `stopLeft`, `warning`, `warningLeft`. Instead of deep learning, we use hand-crafted features (HSV color histograms, HOG) with classical classifiers (Gaussian Bayes, GMM, logistic regression, k-NN, SVM, random forest).

Two aspects set the project apart from typical approaches on this dataset:

- **Leakage-free evaluation.** The data are continuous video, so consecutive frames are nearly identical. All splits are made at the clip level, never at the frame level.
- **Error decomposition and risk.** Errors are split into *color-state errors* (e.g., stop predicted as go) and *shape errors* (e.g., goLeft predicted as go), and a Bayes minimum-risk rule is used to penalize the most dangerous mistakes.

## Repository Structure

```
trafficlights_detection/
├── data/                        # LISA dataset (NOT tracked by git, download manually)
├── cache/                       # cached features (NOT tracked by git, created automatically)
├── data_exploration.ipynb    # Exploratory data analysis
├── baseline.ipynb            # Color histogram + logistic regression baseline
├── .gitignore
└── README.md
```

More notebooks and source files will be added as the project progresses.

## Getting the Data

The dataset is several GB and contains ~44,000 files, so it is **not included in this repository**. Each team member must download it manually.

1. Download the dataset from Kaggle: <https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset>
   - **Option A (browser):** click *Download*, then unzip the archive.
   - **Option B (Kaggle CLI):** after setting up your Kaggle API credentials, run
     ```bash
     kaggle datasets download -d mbornoe/lisa-traffic-light-dataset -p data --unzip
     ```
2. Place the contents in `data/` at the project root so that the layout looks like this:

```
data/
├── Annotations/
│   └── Annotations/             # frameAnnotationsBOX.csv files for every clip
├── daySequence1/
├── daySequence2/
├── dayTrain/
├── nightSequence1/
├── nightSequence2/
├── nightTrain/
├── sample-dayClip6/             # duplicate of dayTrain/dayClip6, not used
└── sample-nightClip1/           # duplicate of nightTrain/nightClip1, not used
```

The folder name and nesting matter: the notebooks look for annotations in `data/Annotations/Annotations/`. The `sample-*` folders are duplicates shipped with the dataset and are excluded by the code; you may delete them.

`data/` and `cache/` are listed in `.gitignore`. Please do not commit any data or cached feature files.

## Environment Setup

We use a conda environment named `LISA` with Python 3.10.

```bash
conda create -n LISA python=3.10
conda activate LISA
python -m pip install numpy pandas matplotlib opencv-python-headless scikit-learn ipykernel
```

Notes:

- Install OpenCV with **pip** as `opencv-python-headless`. This name does not exist on the default conda channel. Do not install `opencv-python` and `opencv-python-headless` at the same time, since they conflict.
- In VS Code or Jupyter, select the `LISA` environment as the notebook kernel.
- The first `import matplotlib.pyplot` may take a minute while matplotlib builds its font cache. This happens only once.

## Notebooks

### `data_exploration.ipynb`

Exploratory data analysis. Each section ends with a **Decision** cell recording the pipeline parameter it determines, and the notebook ends with a summary table to fill in. Run the cells in order, since later cells depend on variables defined earlier (`ann`, `crop`, ...).

Set the data path in the configuration cell if your layout differs:

```python
ROOT = "./data"
```

Sections 1–5 only read the annotation CSVs and run in seconds. Sections 6–7 read image files and take longer.

| Section | What it does | What it decides |
|---|---|---|
| 1. Load annotations | Reads every `frameAnnotationsBOX.csv` into one table `ann`, adding the clip name, train/test split, and day/night flag. | — |
| 2. Integrity checks | Counts invalid boxes, boxes touching the frame border, duplicate rows, and tags outside the six classes (e.g., `goForward`). | Which rows/tags to drop or merge |
| 3. Class distribution | Class counts per clip, per day/night, and per train/test; number of training clips containing each class. | Number of CV folds and fold assignment |
| 4. Box size and shape | Histograms of box height, width, and aspect ratio; per-class size percentiles. | Patch size and minimum box height |
| 5. Temporal redundancy | Frames and boxes per clip; number of physical lights (tracks) and boxes per track. | Frame subsampling step |
| 6. Visual inspection | Indexes all frames, crops lights with a 15% margin, and shows random samples per class (day, night, and the smallest 10% of boxes). | Label-noise estimate; size at which arrows become unreadable |
| 7. Color feature preview | Hue histograms of lit pixels (high saturation and brightness) for go/stop/warning, day vs. night. | Hue encoding (red wraps around 0/179); evidence for GMMs |

### `baseline.ipynb`

The reference result that every later model must beat: a 35-dimensional color-only feature (18-bin hue histogram of lit pixels, saturation and value histograms, lit-pixel fraction) with class-weighted logistic regression.

- Uses only the 18 training clips; the 4 test sequences are reserved for the final evaluation.
- Grouped 6-fold cross-validation by clip (`StratifiedGroupKFold`), so no clip is split between training and validation.
- Reports pooled out-of-fold macro-F1 and balanced accuracy, per-fold macro-F1 (mean ± std), a confusion matrix, and the error decomposition: color error, shape error, 3-class color accuracy, and red→green rate, for all data and for day/night separately.
- Parameters at the top (`K_SUBSAMPLE`, `MIN_H`, `PATCH_WH`) should be updated from the EDA results. Features are cached in `cache/`, so later runs skip image loading; delete the cache file after changing these parameters.

## Project Timeline

| Dates | Milestone |
|---|---|
| Oct 17 | Proposal submitted |
| Oct 18 – Oct 25 | Data exploration, literature review, quick baseline |
| Oct 26 – Nov 8 | Crop pipeline, clip-level splits, feature extraction, evaluation framework |
| Nov 9 – Nov 22 | Generative and discriminative models |
| Nov 23 – Dec 4 | PCA, minimum-risk analysis, error decomposition, day/night analysis |
| Dec 5 – Dec 12 | Final report and code release |

## References

- M. B. Jensen et al., "Vision for looking at traffic lights: Issues, survey, and perspectives," *IEEE Trans. Intelligent Transportation Systems*, vol. 17, no. 7, pp. 1800–1815, 2016.
- M. P. Philipsen et al., "Traffic light detection: A learning algorithm and evaluations on challenging dataset," *Proc. IEEE ITSC*, 2015, pp. 2341–2345.