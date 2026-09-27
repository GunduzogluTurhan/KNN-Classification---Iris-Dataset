# Iris Classification with K-Nearest Neighbors (from scratch)

A K-Nearest Neighbors (KNN) classifier for the classic Iris dataset, written
**from scratch in Python**. The KNN algorithm, z-score standardization, data
splitting and all evaluation metrics are implemented by hand; `pandas` is used
only to read the CSV file and `matplotlib` only for the plots.

## Repository contents

| File | Description |
|---|---|
| `Classification.py` | The complete pipeline: data checks, splitting, scaling, KNN, K selection, evaluation, experiments and plots |
| `Iris.csv` | The Iris dataset (150 samples, 4 features, 3 classes) |

## Dataset

- **150 samples**, 50 from each species: *Iris-setosa*, *Iris-versicolor*, *Iris-virginica*
- **4 numerical features (cm):** `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`
- An `Id` column (row identifier only, removed before modeling)

**Data correction:** The widely shared version of this CSV contains two known
errors, documented by the UCI Machine Learning Repository. They were corrected
so the file matches Fisher's original 1936 data:

| Id | Original CSV | Corrected |
|---|---|---|
| 35 | `4.9, 3.1, 1.5, 0.1` | `4.9, 3.1, 1.5, 0.2` |
| 38 | `4.9, 3.1, 1.5, 0.1` | `4.9, 3.6, 1.4, 0.1` |

After the correction the data contains no missing values. It has one pair of
rows with identical measurements (Id 102 and Id 143, both *Iris-virginica*).
This pair also exists in Fisher's original data. They are two different
flowers, so both rows are kept.

## Method

1. **Data quality checks:** missing values, duplicates (checked *after*
   removing `Id`), identical measurements with conflicting labels, class
   balance.
2. **Stratified split (seed 42):** 60% train / 20% validation / 20% test,
   which gives 30 / 10 / 10 samples per class.
3. **Standardization:** `z = (x - mean) / std`. The mean and std are computed
   from the **training set only** and then applied to the validation and test
   sets, so no information leaks from them.
4. **KNN:** Euclidean distance with majority voting.
   - *Distance ties:* if other training samples are exactly as far away as the
     K-th neighbor, all of them take part in the vote. This makes the result
     independent of the order of the training data.
   - *Vote ties:* the class whose neighbors have the smaller total distance wins.
5. **K selection:** K ∈ {1, 3, 5, 7, 9, 11, 13, 15} is evaluated on the
   validation set. The K with the highest accuracy is chosen; if several K
   values tie, the smallest one is chosen.
6. **Final evaluation:** the test set is used once, with the selected K.
   The script reports accuracy, the confusion matrix, per-class
   precision / recall / F1, and macro averages.
7. **Scaling experiment:** the same pipeline is run without scaling.
8. **Stability check:** the whole pipeline is repeated on 100 different random
   stratified splits, because with only 30 test samples a single split can be
   misleading.

## Results

Validation accuracy for each K:

| K | 1 | 3 | 5 | **7** | 9 | 11 | 13 | 15 |
|---|---|---|---|---|---|---|---|---|
| Accuracy | 0.9000 | 0.9333 | 0.9333 | **0.9667** | 0.9333 | 0.9333 | 0.9000 | 0.8667 |

**Selected K = 7. Test accuracy = 0.9667 (29 / 30).**

Confusion matrix on the test set (rows = actual class, columns = predicted class):

| | setosa | versicolor | virginica |
|---|---|---|---|
| **setosa** | 10 | 0 | 0 |
| **versicolor** | 0 | 10 | 0 |
| **virginica** | 0 | 1 | 9 |

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Iris-setosa | 1.0000 | 1.0000 | 1.0000 | 10 |
| Iris-versicolor | 0.9091 | 1.0000 | 0.9524 | 10 |
| Iris-virginica | 1.0000 | 0.9000 | 0.9474 | 10 |
| **Macro average** | **0.9697** | **0.9667** | **0.9666** | 30 |

*Iris-setosa* is always classified correctly. The only error is one
*Iris-virginica* sample (Id 150) predicted as *Iris-versicolor*. These two
species overlap in feature space, as the scatter plots show.

### Effect of feature scaling

| Setting | Best K (seed 42) | Test acc. (seed 42) | Mean test acc. (100 splits) | Std |
|---|---|---|---|---|
| With standardization | 7 | 0.9667 | 0.9480 | 0.0386 |
| Without scaling | 1 | 1.0000 | 0.9580 | 0.0360 |

On this dataset, standardization **does not improve** accuracy. It is slightly
worse on average: better in 16 splits, worse in 44, equal in 40. All four
features already share the same unit (cm) with ranges of similar magnitude.
Standardization increases the relative weight of `SepalWidthCm`, the feature
with the smallest spread and the weakest class separation. Scaling is still
kept in the main pipeline because it is the standard, safe choice for
distance-based methods on data in general.

## How to run

Requirements: Python 3.8+ with `pandas` and `matplotlib` (tested with
pandas 2.3 / 3.0 and matplotlib 3.9 / 3.10).

```bash
pip install pandas matplotlib
python Classification.py
```

`Iris.csv` must be in the same folder as `Classification.py`. The script
prints every step to the console and saves four figures to a `figures/`
folder:

- `k_optimization.png`: validation accuracy for each K
- `confusion_matrix.png`: test-set confusion matrix
- `feature_distributions.png`: histograms of each feature by class
- `feature_relationships.png`: sepal and petal scatter plots by class

The whole run takes a few seconds. All results are reproducible because the
random seeds are fixed.

## References

1. Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems.
   *Annals of Eugenics*, 7(2), 179–188.
2. Fisher, R. A. (1936). Iris [Dataset]. UCI Machine Learning Repository.
   https://doi.org/10.24432/C56C76
3. Cover, T. M., & Hart, P. E. (1967). Nearest neighbor pattern classification.
   *IEEE Transactions on Information Theory*, 13(1), 21–27.
