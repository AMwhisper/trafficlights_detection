# Traffic Light State Recognition with Classical Machine Learning

EECE 5644 (Machine Learning and Pattern Recognition), Fall 2026 — Term Project
**Team:** Jingqi Mei, Andrew Chen

## Overview

This project classifies cropped traffic lights from the [LISA Traffic Light Dataset](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset) into six classes: `go`, `goLeft`, `stop`, `stopLeft`, `warning`, `warningLeft`. Instead of deep learning, we use hand-crafted features (color histograms, position, red dominance, HOG) with classical classifiers (Gaussian Bayes, GMM, logistic regression, k-NN, SVM, random forest).

Two aspects set the project apart from typical approaches on this dataset:

- **Leakage-free evaluation.** The data are continuous video, so consecutive frames are nearly identical. All splits are made at the clip level, never at the frame level.
- **Error decomposition and risk.** Errors are split into *color-state errors* (e.g., stop predicted as go) and *shape errors* (e.g., goLeft predicted as go), and a Bayes minimum-risk rule is used to penalize the most dangerous mistakes.

## Repository Structure

```
trafficlights_detection/
├── data/                      # LISA dataset (NOT tracked by git, download manually)
├── cache/                     # cached patches and features (NOT tracked by git)
├── splits/
│   └── annotations_clean.csv  # cleaned labels + fixed CV folds (tracked by git)
├── data_exploration.ipynb     # Exploratory data analysis
├── baseline.ipynb             # Color histogram + logistic regression baseline
├── color_position.ipynb       # Feature experiment: color + position
├── hog.ipynb                  # Main feature pipeline: all features, fixed folds, ablation, export
├── DATA_PACKAGE.md            # Description of the shared features and folds
├── .gitignore
└── README.md
```

Other intermediate feature-experiment notebooks are superseded by `hog.ipynb`, which reproduces all feature sets from one cache. More notebooks and source files (e.g. `src/evaluate.py`) will be added for the modeling stage.

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

`data/` and `cache/` are listed in `.gitignore`. Please do not commit any data or cached files. Cached files that need to be shared are exchanged via a shared drive (see `DATA_PACKAGE.md`).

**If you only work on models**, you do not need the raw dataset: the shared feature file plus `splits/annotations_clean.csv` are enough.

## Environment Setup

We use a conda environment named `LISA` with Python 3.10.

```bash
conda create -n LISA python=3.10
conda activate LISA
python -m pip install numpy pandas matplotlib opencv-python-headless scikit-learn scikit-image ipykernel
```

Notes:

- Install OpenCV with **pip** as `opencv-python-headless`. This name does not exist on the default conda channel. Do not install `opencv-python` and `opencv-python-headless` at the same time, since they conflict.
- In VS Code or Jupyter, select the `LISA` environment as the notebook kernel.
- The first `import matplotlib.pyplot` may take a minute while matplotlib builds its font cache. This happens only once.

## Frozen Pipeline Parameters

All shared results use the following settings. Changing any of them invalidates the shared features, folds, and results.

| Parameter | Value | Meaning |
|---|---|---|
| `K_SUBSAMPLE` | 5 | Keep every 5th frame of each clip |
| `MIN_H` | 5 | Drop boxes shorter than 5 px |
| `PATCH_WH` | (16, 32) | Resized patch size (width × height) |
| `N_FOLDS` | 5 | Clip-level folds, one night clip per fold |
| HOG | 9 orientations, 4 × 4 cells, 2 × 2 blocks | Computed on the V channel |

## Notebooks

### `data_exploration.ipynb`

Exploratory data analysis. Each section ends with a **Decision** cell recording the pipeline parameter it determines, and the notebook ends with a summary table to fill in. Run the cells in order, since later cells depend on variables defined earlier (`ann`, `crop`, ...).

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

The first reference result: a 35-dimensional color-only feature (18-bin hue histogram of lit pixels, saturation and value histograms, lit-pixel fraction) with class-weighted logistic regression. It uses only the 18 training clips and grouped cross-validation by clip, and reports pooled out-of-fold metrics and the error decomposition.

This notebook uses the earlier 6-fold `StratifiedGroupKFold` split. Its numbers are kept for the record, but **the official results use the fixed 5-fold split in `hog.ipynb`**, where the baseline is reproduced as the "color" feature set.

### `color_position.ipynb`

Feature experiment that adds 4 position features (brightness share of the top/middle/bottom thirds, height of the brightest pixels) to the color features. It showed that position resolves most red/yellow confusion. Superseded by `hog.ipynb`.

### `hog.ipynb`

The main feature pipeline, producing all shared data for the modeling stage.

1. **Load and filter** the training annotations with the frozen parameters.
2. **Crop and cache patches**: each frame is read once, every box is cropped and resized to 16 × 32, and all patches are saved to `cache/patches_*.npz`. Later feature changes take seconds instead of re-reading frames.
3. **Feature functions**: color (35), position (4), red dominance (2), and HOG (756), for 797 features in total.
4. **Fixed 5-fold assignment**: one night clip per fold, day clips assigned by a random search that balances fold sizes and class presence (using labels only, never model results). Stored in `ann["fold"]`.
5. **Ablation**: logistic regression on five feature sets (color, + position, + red, HOG only, all), with pooled OOF metrics, per-fold macro-F1, and the error decomposition.
6. **Analysis**: confusion matrix, day/night breakdown, per-clip error table, and visualization of misclassified patches.
7. **Export**: writes `splits/annotations_clean.csv` and `cache/features_color_pos_red_hog.npz` for the teammate. See `DATA_PACKAGE.md` for the file formats.

## Results So Far

Logistic regression (`StandardScaler`, `class_weight="balanced"`), pooled out-of-fold over the fixed 5 folds of the 18 training clips:

| Features | Macro-F1 | Balanced acc. | Color error | Shape error | Red → green |
|---|---|---|---|---|---|
| Color | 0.390 | 0.463 | 12.4% | 26.6% | 6.2% |
| + Position | 0.468 | 0.511 | 4.7% | 28.2% | 3.6% |
| + Red dominance | 0.487 | 0.534 | 3.2% | 28.0% | 2.0% |
| HOG only | **0.611** | **0.640** | 2.8% | 14.2% | 1.8% |
| All features | 0.604 | 0.599 | **1.2%** | **14.1%** | **0.5%** |

Key findings so far:

- **Position resolves red/yellow confusion.** Red and amber have adjacent hues and are often confused in daylight; the vertical position of the lit lamp separates them.
- **Background causes shortcut learning.** In some daytime clips, saturated blue sky passes the lit-pixel threshold and is confused with blue-green lamps, pulling red arrows toward `goLeft`. The red-dominance feature is immune to this, because sky is never red-dominant. Cropping the box center instead was a negative result: it removed sky but also damaged arrow-shape cues.
- **HOG fixes most shape errors** and also infers color implicitly from lamp position. It also makes results more stable across folds.
- **Some errors are not recoverable from pixels**, e.g. LED lights captured while off (flicker) and heavily overexposed or color-shifted frames in `dayClip13`.

The four test sequences have not been used yet.

## Project Timeline

| Dates | Milestone | Status |
|---|---|---|
| Oct 17 | Proposal submitted | Done |
| Oct 18 – Oct 25 | Data exploration, literature review, quick baseline | Done |
| Oct 26 – Nov 8 | Crop pipeline, clip-level splits, feature extraction, evaluation framework | Features and folds done |
| Nov 9 – Nov 22 | Generative and discriminative models | Next |
| Nov 23 – Dec 4 | PCA, minimum-risk analysis, error decomposition, day/night analysis, final test evaluation | |
| Dec 5 – Dec 12 | Final report and code release | |

## References

- M. B. Jensen et al., "Vision for looking at traffic lights: Issues, survey, and perspectives," *IEEE Trans. Intelligent Transportation Systems*, vol. 17, no. 7, pp. 1800–1815, 2016.
- M. P. Philipsen et al., "Traffic light detection: A learning algorithm and evaluations on challenging dataset," *Proc. IEEE ITSC*, 2015, pp. 2341–2345.
- N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," *Proc. IEEE CVPR*, 2005, pp. 886–893.