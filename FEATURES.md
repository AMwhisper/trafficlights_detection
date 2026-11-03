# Data Package for the Modeling Stage

This document describes the features, labels, and cross-validation folds shared between team members for the modeling stage of the project. All files are produced by `hog.ipynb`.

Everything is computed from the **18 training clips only**. The 4 test sequences (`daySequence1/2`, `nightSequence1/2`) are untouched and reserved for the final evaluation.

## Files

| File | Location | Contents |
|---|---|---|
| `annotations_clean.csv` | `splits/` (tracked in git) | One row per sample (10,392 rows): labels, clip, day/night flag, box, fold |
| `features_color_pos_red_hog.npz` | `cache/` (shared drive, not in git) | `X` (10392 × 797, float32), `feature_names`, `params` |
| `patches_k5_h5_16x32.npz` | `cache/` (shared drive, optional) | `P` (10392 × 32 × 16 × 3, RGB uint8): the resized crops, useful for inspecting errors |

> **The row order of the CSV, `X`, and `P` is identical.** Do not sort or filter one of them without applying the same operation to the others.

### Columns of `annotations_clean.csv`

| Column | Description |
|---|---|
| `frame` | Frame file name, e.g. `dayClip1--00120.jpg` |
| `clip` | Source clip, e.g. `dayTrain/dayClip1` |
| `isNight` | `True` for night clips |
| `tag` | Class label (one of the six classes below) |
| `x1`, `y1`, `x2`, `y2` | Bounding box in the original 1280 × 960 frame |
| `w`, `h` | Box width and height (pixels) |
| `fold` | Cross-validation fold (0–4) |
| `color` | Signal color derived from `tag`: green, red, or yellow |

## How the Samples Were Built

1. Load all `frameAnnotationsBOX.csv` files of the 18 training clips.
2. Keep the six classes `go`, `goLeft`, `stop`, `stopLeft`, `warning`, `warningLeft`; drop other tags (e.g. `goForward`).
3. Remove invalid boxes, boxes shorter than 5 px, and duplicate annotations.
4. Keep every 5th frame of each clip to reduce temporal redundancy.
5. Crop each box tightly and resize it to 16 × 32 (width × height) with `cv2.INTER_AREA`.

These parameters (`K_SUBSAMPLE=5`, `MIN_H=5`, `PATCH_WH=(16, 32)`) are **frozen**. Changing them would invalidate all shared results.

### Class distribution

| | go | goLeft | stop | stopLeft | warning | warningLeft |
|---|---|---|---|---|---|---|
| Day | 2,778 | 170 | 3,021 | 1,398 | 158 | 59 |
| Night | 1,825 | 76 | 657 | 148 | 101 | 1 |
| **Total** | 4,603 | 246 | 3,678 | 1,546 | 259 | 60 |

`warningLeft` has only 60 samples, so results for this class are unstable.

## Feature Columns

| Columns | Group | Dims | Description |
|---|---|---|---|
| 0–34 | Color | 35 | 18-bin hue histogram of lit pixels (S ≥ 80, V ≥ 120), 8-bin saturation and 8-bin value histograms, lit-pixel fraction |
| 35–38 | Position | 4 | Brightness share of the top, middle, and bottom thirds; mean height of the brightest 10% of pixels |
| 39–40 | Red dominance | 2 | Fraction of pixels with R − max(G, B) > 40; maximum red dominance |
| 41–796 | HOG | 756 | HOG on the V channel: 9 orientations, 4 × 4-pixel cells, 2 × 2-cell blocks, L2-Hys normalization |

Column names are stored in `feature_names`. Useful slices:

- `X[:, :35]` color only
- `X[:, :39]` color + position
- `X[:, :41]` color + position + red
- `X[:, 41:]` HOG only
- `X` all features

## Cross-Validation Protocol

**Always use the `fold` column. Do not create new splits.**

- 5 folds; each fold contains whole clips, so no clip is split between training and validation.
- Each fold contains exactly one night clip and samples from all six classes.
- The assignment was chosen by a random search that only used class counts and fold sizes, never model results.

For each fold `k`: train on `fold != k`, validate on `fold == k`. Report:

- **Pooled out-of-fold (OOF) metrics**: every sample is predicted by a model that never saw its clip.
- **Per-fold macro-F1** (mean ± std), averaged only over the classes present in that fold.

Any preprocessing that learns from data (standardization, PCA, ...) must be **fit inside each training fold only**. Fitting it on all samples leaks validation information.

## Metrics

| Metric | Definition |
|---|---|
| **Macro-F1** (primary) | Unweighted mean of per-class F1 over the six classes |
| Balanced accuracy | Mean per-class recall |
| Color error | Predicted color ≠ true color |
| Shape error | Correct color, but wrong arrow/circle type |
| Red → green rate | Among true red lights, fraction predicted as any green class |

Color error + shape error = total six-class error. The reference implementations (`run_cv`, `summarize`, `error_decomposition`) are in section 6 of `hog.ipynb`. They should be moved into `src/evaluate.py` so that both team members compute metrics with exactly the same code.

## Current Results

Logistic regression with `StandardScaler` and `class_weight="balanced"`, pooled OOF over the 5 folds:

| Features | Macro-F1 | Balanced acc. | Color error | Shape error | Red → green |
|---|---|---|---|---|---|
| Color | 0.390 | 0.463 | 12.4% | 26.6% | 6.2% |
| + Position | 0.468 | 0.511 | 4.7% | 28.2% | 3.6% |
| + Red dominance | 0.487 | 0.534 | 3.2% | 28.0% | 2.0% |
| HOG only | **0.611** | **0.640** | 2.8% | 14.2% | 1.8% |
| All features | 0.604 | 0.599 | **1.2%** | **14.1%** | **0.5%** |

These are the reference numbers that new models should be compared against.

## Notes for the Generative Models

- 797 dimensions are too many for reliable full-covariance estimates. Reduce dimensionality with PCA first, mainly on the HOG part, fitted inside each training fold.
- The Gaussian class-conditional covariances will likely need regularization (shrinkage or diagonal covariance).
- Keep the predicted **posterior probabilities**, not just the labels. They are needed for the minimum-risk analysis.

## Loading Example

```python
import numpy as np
import pandas as pd

ann = pd.read_csv("splits/annotations_clean.csv")
d = np.load("cache/features_color_pos_red_hog.npz")
X, feature_names = d["X"], d["feature_names"]

y = ann["tag"].to_numpy()
fold = ann["fold"].to_numpy()

for k in range(5):
    tr, va = fold != k, fold == k
    # fit preprocessing + model on X[tr], y[tr]; predict X[va]
```

To view a crop: `P = np.load("cache/patches_k5_h5_16x32.npz")["P"]`, then `plt.imshow(P[i])`.