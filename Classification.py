"""
K-Nearest Neighbors (KNN) classification of the Iris dataset.

Everything (KNN, standardization, metrics) is implemented from scratch;
pandas is used only for loading/inspecting the CSV and matplotlib for plots.

Pipeline
    1. Load and validate Iris.csv
    2. Data quality checks (missing values, duplicates, class balance)
    3. Stratified train / validation / test split (60% / 20% / 20%)
    4. Standardization (z-score) fitted on the TRAINING set only
    5. K selection on the validation set
    6. Final evaluation on the untouched test set
    7. Extra experiment: the same pipeline without scaling
    8. Stability check over 100 different random splits
    9. Figures (saved to ./figures)

Run:  python Classification.py
"""

import math
import os
import random
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


# =========================
# 0. SETTINGS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Iris.csv")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")

RANDOM_SEED = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2          # the remaining 20% is the test set
K_VALUES = [1, 3, 5, 7, 9, 11, 13, 15]
DISTANCE_TOLERANCE = 1e-9    # distances closer than this are treated as equal

FEATURE_COLUMNS = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
TARGET_COLUMN = "Species"


def save_and_show(filename):
    """Save the current figure to the figures folder and show it if possible."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename), dpi=150)
    # Without a display matplotlib falls back to the "agg" backend; only
    # open a window when an interactive backend is available.
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close()


# =========================
# 1. LOAD DATASET FROM CSV
# =========================

df = pd.read_csv(DATA_PATH)

required_columns = ["Id"] + FEATURE_COLUMNS + [TARGET_COLUMN]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns in {DATA_PATH}: {missing_columns}")

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset shape (rows, columns):")
print(df.shape)


# =========================
# 2. DATA QUALITY CHECKS AND CLEANING
# =========================

print("\nMissing values per column:")
print(df.isnull().sum())
if df.isnull().values.any():
    raise ValueError("The dataset contains missing values.")

# The Id column is only a row identifier and carries no information about the
# flower, so it is removed. It must be removed BEFORE checking for duplicates:
# every Id is unique, so a duplicate check that includes Id can never find one.
df = df.drop("Id", axis=1)

duplicate_mask = df.duplicated(keep=False)
n_duplicates = int(df.duplicated().sum())
print(f"\nDuplicate rows (after removing Id): {n_duplicates}")
if n_duplicates > 0:
    print("Rows with identical measurements and label:")
    print(df[duplicate_mask])
    print(
        "Note: these are two different flowers that happen to have identical\n"
        "measurements (they also appear in Fisher's original 1936 data), so\n"
        "they are real observations and are kept in the dataset."
    )

# Identical measurements with DIFFERENT labels would be a real inconsistency
conflicts = df.groupby(FEATURE_COLUMNS)[TARGET_COLUMN].nunique()
n_conflicts = int((conflicts > 1).sum())
print(f"Identical measurements with conflicting labels: {n_conflicts}")
if n_conflicts > 0:
    raise ValueError("Some samples have identical features but different labels.")

print("\nClass distribution:")
print(df[TARGET_COLUMN].value_counts().sort_index())

print("\nDescriptive statistics (cm):")
print(df[FEATURE_COLUMNS].describe().round(3))

classes = sorted(df[TARGET_COLUMN].unique())


# =========================
# 3. TRAIN / VALIDATION / TEST SPLIT (STRATIFIED)
# =========================
# The split is stratified: every class is shuffled and divided 60/20/20
# separately, so each subset keeps the original 1:1:1 class balance.
# The split is done on the RAW data; scaling is fitted afterwards on the
# training set only, so no information from validation/test data leaks
# into the model.

def stratified_split(X, y, train_ratio, val_ratio, seed):
    rng = random.Random(seed)
    train, val, test = [], [], []

    for cls in sorted(set(y)):
        samples = [(features, label) for features, label in zip(X, y) if label == cls]
        rng.shuffle(samples)

        n = len(samples)
        train_end = round(n * train_ratio)
        val_end = train_end + round(n * val_ratio)

        train += samples[:train_end]
        val += samples[train_end:val_end]
        test += samples[val_end:]

    # Mix the classes inside each subset
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    def unpack(data):
        return [item[0] for item in data], [item[1] for item in data]

    return unpack(train), unpack(val), unpack(test)


X_raw = df[FEATURE_COLUMNS].values.tolist()
y = df[TARGET_COLUMN].values.tolist()

(X_train_raw, y_train), (X_val_raw, y_val), (X_test_raw, y_test) = stratified_split(
    X_raw, y, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED
)

print("\nTrain size:", len(X_train_raw), dict(sorted(Counter(y_train).items())))
print("Validation size:", len(X_val_raw), dict(sorted(Counter(y_val).items())))
print("Test size:", len(X_test_raw), dict(sorted(Counter(y_test).items())))


# =========================
# 4. FEATURE SCALING (STANDARDIZATION / Z-SCORE)
# =========================
# KNN is distance-based, so features are put on a common scale:
#     z = (x - mean) / std
# mean and std are computed from the TRAINING set only and then applied
# unchanged to the validation and test sets.

class StandardScaler:

    def fit(self, X):
        n_features = len(X[0])
        self.means = []
        self.stds = []

        for col in range(n_features):
            values = [row[col] for row in X]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            std = math.sqrt(variance)

            self.means.append(mean)
            self.stds.append(std if std > 0 else 1.0)   # avoid division by zero

        return self

    def transform(self, X):
        return [
            [(row[i] - self.means[i]) / self.stds[i] for i in range(len(row))]
            for row in X
        ]


scaler = StandardScaler().fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

print("\nScaling parameters (computed from the training set only):")
for name, mean, std in zip(FEATURE_COLUMNS, scaler.means, scaler.stds):
    print(f"{name:<14} mean = {mean:.4f} | std = {std:.4f}")


# =========================
# 5. KNN CLASSIFIER
# =========================

class KNNClassifier:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        # KNN is a lazy learner: "training" only stores the data
        self.X_train = X_train
        self.y_train = y_train
        return self

    def euclidean_distance(self, x1, x2):
        total = 0
        for i in range(len(x1)):
            total += (x1[i] - x2[i]) ** 2
        return math.sqrt(total)

    def predict_one(self, x):
        distances = []

        for i in range(len(self.X_train)):
            distance = self.euclidean_distance(x, self.X_train[i])
            distances.append((distance, self.y_train[i]))

        distances.sort(key=lambda item: item[0])

        # Neighbor selection with tie handling.
        # Iris values have one decimal place, so several training samples are
        # often at exactly the same distance. If the K-th and (K+1)-th
        # neighbors are equally far, picking only one of them would depend on
        # the arbitrary order of the training data. Therefore every sample
        # tied with the K-th neighbor is included in the vote.
        # A small tolerance absorbs floating-point rounding (e.g. 5.1-4.9).
        kth_distance = distances[self.k - 1][0]
        neighbors = [(d, label) for d, label in distances if d <= kth_distance + DISTANCE_TOLERANCE]

        # Majority vote
        votes = Counter(label for _, label in neighbors)
        top_count = max(votes.values())
        tied_classes = [label for label, count in votes.items() if count == top_count]

        if len(tied_classes) == 1:
            return tied_classes[0]

        # Vote tie: the class whose neighbors are closer in total wins;
        # if that is also equal, the alphabetically first class is chosen.
        def total_distance(label):
            return sum(d for d, lbl in neighbors if lbl == label)

        return min(tied_classes, key=lambda label: (round(total_distance(label), 9), label))

    def predict(self, X):
        return [self.predict_one(x) for x in X]


# =========================
# 6. EVALUATION FUNCTIONS
# =========================

def accuracy_score(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)


def confusion_matrix_manual(y_true, y_pred, classes):
    # rows = actual class, columns = predicted class
    matrix = []
    for actual_class in classes:
        row = []
        for predicted_class in classes:
            count = sum(
                1 for true, pred in zip(y_true, y_pred)
                if true == actual_class and pred == predicted_class
            )
            row.append(count)
        matrix.append(row)
    return matrix


def precision_recall_f1(y_true, y_pred, classes):
    results = {}

    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        results[cls] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": tp + fn,
        }

    return results


def select_best_k(accuracy_by_k):
    """Highest validation accuracy; on a tie the smallest K is chosen."""
    best_accuracy = max(accuracy_by_k.values())
    tied = [k for k, acc in accuracy_by_k.items() if acc == best_accuracy]
    return min(tied), tied, best_accuracy


# =========================
# 7. K OPTIMIZATION (VALIDATION SET)
# =========================

val_results = {}

for k in K_VALUES:
    model = KNNClassifier(k=k).fit(X_train, y_train)
    val_predictions = model.predict(X_val)
    val_results[k] = accuracy_score(y_val, val_predictions)

print("\nK Optimization Results on Validation Set:")
for k, acc in val_results.items():
    correct = round(acc * len(y_val))
    print(f"K = {k:<2} | Validation Accuracy = {acc:.4f} ({correct}/{len(y_val)})")

best_k, tied_ks, best_val_accuracy = select_best_k(val_results)

print(f"\nHighest validation accuracy: {best_val_accuracy:.4f}")
if len(tied_ks) > 1:
    print(f"K values tied at this accuracy: {tied_ks}")
    print("Tie-break rule: the smallest tied K is selected.")
print(f"Best K selected from Validation Set: {best_k}")


# =========================
# 8. FINAL MODEL (TEST SET)
# =========================
# The final model uses the training set and the K chosen above.
# The test set is used exactly once, here.

final_model = KNNClassifier(k=best_k).fit(X_train, y_train)
final_predictions = final_model.predict(X_test)

print("\nFinal Test Predictions:")
for real, pred in zip(y_test, final_predictions):
    mark = "" if real == pred else "   <-- misclassified"
    print(f"Real: {real:<15} | Predicted: {pred:<15}{mark}")

final_accuracy = accuracy_score(y_test, final_predictions)
n_correct = sum(1 for t, p in zip(y_test, final_predictions) if t == p)

print(f"\nFinal Accuracy on Test Set: {final_accuracy:.4f} ({n_correct}/{len(y_test)})")


# =========================
# 9. CONFUSION MATRIX
# =========================

cm = confusion_matrix_manual(y_test, final_predictions, classes)

print("\nConfusion Matrix (rows = actual class, columns = predicted class):")
print(" " * 17 + "".join(f"{c:>17}" for c in classes))
for cls, row in zip(classes, cm):
    print(f"{cls:<17}" + "".join(f"{v:>17}" for v in row))


# =========================
# 10. PRECISION, RECALL, F1-SCORE
# =========================

metrics = precision_recall_f1(y_test, final_predictions, classes)

print("\nClassification Metrics (test set):")
print(f"{'Class':<17}{'Precision':>10}{'Recall':>10}{'F1-Score':>10}{'Support':>9}")
for cls, values in metrics.items():
    print(
        f"{cls:<17}{values['precision']:>10.4f}{values['recall']:>10.4f}"
        f"{values['f1-score']:>10.4f}{values['support']:>9}"
    )

macro_precision = sum(m["precision"] for m in metrics.values()) / len(classes)
macro_recall = sum(m["recall"] for m in metrics.values()) / len(classes)
macro_f1 = sum(m["f1-score"] for m in metrics.values()) / len(classes)
print(
    f"{'Macro average':<17}{macro_precision:>10.4f}{macro_recall:>10.4f}"
    f"{macro_f1:>10.4f}{len(y_test):>9}"
)


# =========================
# 11. EFFECT OF FEATURE SCALING (EXTRA EXPERIMENT)
# =========================
# The same split, K range and selection rule, but WITHOUT standardization.
# This measures the effect of scaling on this dataset instead of assuming it.

def run_pipeline(X_tr, y_tr, X_va, y_va, X_te, y_te, scale):
    """Select K on the validation set and return (best K, val acc, test acc)."""
    if scale:
        split_scaler = StandardScaler().fit(X_tr)
        X_tr, X_va, X_te = (split_scaler.transform(X) for X in (X_tr, X_va, X_te))

    accuracy_by_k = {
        k: accuracy_score(y_va, KNNClassifier(k=k).fit(X_tr, y_tr).predict(X_va))
        for k in K_VALUES
    }
    k_best, _, val_acc = select_best_k(accuracy_by_k)
    test_acc = accuracy_score(y_te, KNNClassifier(k=k_best).fit(X_tr, y_tr).predict(X_te))
    return k_best, val_acc, test_acc


raw_val_results = {}
for k in K_VALUES:
    model = KNNClassifier(k=k).fit(X_train_raw, y_train)
    raw_val_results[k] = accuracy_score(y_val, model.predict(X_val_raw))

raw_best_k, raw_tied_ks, raw_best_val = select_best_k(raw_val_results)
raw_model = KNNClassifier(k=raw_best_k).fit(X_train_raw, y_train)
raw_test_accuracy = accuracy_score(y_test, raw_model.predict(X_test_raw))

print("\nEffect of Feature Scaling (same split, same K selection rule):")
print(f"{'Setting':<22}{'Best K':>7}{'Val Acc':>10}{'Test Acc':>10}")
print(f"{'With standardization':<22}{best_k:>7}{best_val_accuracy:>10.4f}{final_accuracy:>10.4f}")
print(f"{'Without scaling':<22}{raw_best_k:>7}{raw_best_val:>10.4f}{raw_test_accuracy:>10.4f}")


# =========================
# 12. STABILITY CHECK OVER MANY RANDOM SPLITS
# =========================
# With only 30 test samples, one misclassification changes accuracy by
# 3.3 points, so a single split can be misleading. The whole pipeline is
# repeated on N_REPEATS different stratified splits (seeds 0..N_REPEATS-1),
# with and without scaling, and the average test accuracy is reported.

N_REPEATS = 100

scaled_accs, raw_accs, chosen_ks = [], [], []
for seed in range(N_REPEATS):
    (Xa, ya), (Xb, yb), (Xc, yc) = stratified_split(X_raw, y, TRAIN_RATIO, VAL_RATIO, seed)
    k_s, _, acc_s = run_pipeline(Xa, ya, Xb, yb, Xc, yc, scale=True)
    _, _, acc_r = run_pipeline(Xa, ya, Xb, yb, Xc, yc, scale=False)
    scaled_accs.append(acc_s)
    raw_accs.append(acc_r)
    chosen_ks.append(k_s)


def mean_std(values):
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    return mean, std


scaled_mean, scaled_std = mean_std(scaled_accs)
raw_mean, raw_std = mean_std(raw_accs)
scaled_wins = sum(1 for s, r in zip(scaled_accs, raw_accs) if s > r)
raw_wins = sum(1 for s, r in zip(scaled_accs, raw_accs) if r > s)

print(f"\nStability Check over {N_REPEATS} Random Stratified Splits (test accuracy):")
print(f"{'Setting':<22}{'Mean':>8}{'Std':>8}{'Min':>8}{'Max':>8}")
print(f"{'With standardization':<22}{scaled_mean:>8.4f}{scaled_std:>8.4f}"
      f"{min(scaled_accs):>8.4f}{max(scaled_accs):>8.4f}")
print(f"{'Without scaling':<22}{raw_mean:>8.4f}{raw_std:>8.4f}"
      f"{min(raw_accs):>8.4f}{max(raw_accs):>8.4f}")
print(f"Splits where standardization was better: {scaled_wins}, "
      f"worse: {raw_wins}, equal: {N_REPEATS - scaled_wins - raw_wins}")
print("How often each K was selected (with standardization):",
      dict(sorted(Counter(chosen_ks).items())))

if scaled_mean > raw_mean:
    print("Conclusion: on average, standardization improved KNN on this dataset.")
elif scaled_mean < raw_mean:
    print(
        "Conclusion: on average, standardization did NOT improve KNN on this dataset.\n"
        "All four Iris features are already in the same unit (cm) and their ranges\n"
        "are of the same order of magnitude (2.4 cm to 5.9 cm).\n"
        "Standardization enlarges the relative weight of SepalWidthCm (the feature\n"
        "with the smallest spread and the weakest class separation), which slightly\n"
        "reduces accuracy. Scaling is still kept in the main pipeline because it is\n"
        "the standard, safe choice for distance-based methods on general data."
    )
else:
    print("Conclusion: standardization made no difference on average for this dataset.")


# =========================
# 13. FIGURES
# =========================

# 13.1 K vs validation accuracy
plt.figure(figsize=(7, 4.5))
plt.plot(list(val_results.keys()), list(val_results.values()), marker="o")
plt.axvline(best_k, linestyle="--", color="gray", label=f"Selected K = {best_k}")
plt.xticks(K_VALUES)
plt.xlabel("K Value")
plt.ylabel("Validation Accuracy")
plt.title("K Optimization on Validation Set")
plt.grid(True)
plt.legend()
save_and_show("k_optimization.png")

# 13.2 Confusion matrix heatmap
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.colorbar(label="Number of samples")
plt.title(f"Confusion Matrix (Test Set, K = {best_k})")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks(range(len(classes)), classes, rotation=45)
plt.yticks(range(len(classes)), classes)
max_value = max(max(row) for row in cm)
for i in range(len(classes)):
    for j in range(len(classes)):
        color = "white" if cm[i][j] > max_value / 2 else "black"
        plt.text(j, i, cm[i][j], ha="center", va="center", color=color)
save_and_show("confusion_matrix.png")

# 13.3 Feature distributions per class (raw values in cm)
FEATURE_LABELS = {
    "SepalLengthCm": "Sepal Length (cm)",
    "SepalWidthCm": "Sepal Width (cm)",
    "PetalLengthCm": "Petal Length (cm)",
    "PetalWidthCm": "Petal Width (cm)",
}

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, feature in zip(axes.flat, FEATURE_COLUMNS):
    # Same bin edges for all classes so the bars are directly comparable
    low, high = df[feature].min(), df[feature].max()
    n_bins = 15
    bin_edges = [low + (high - low) * i / n_bins for i in range(n_bins)] + [high]
    for cls in classes:
        ax.hist(df.loc[df[TARGET_COLUMN] == cls, feature], bins=bin_edges, alpha=0.6, label=cls)
    ax.set_title(f"Distribution of {FEATURE_LABELS[feature]}")
    ax.set_xlabel(FEATURE_LABELS[feature])
    ax.set_ylabel("Frequency")
    ax.grid(True)
axes[0, 0].legend()
save_and_show("feature_distributions.png")

# 13.4 Relationships between features (raw values in cm)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
pairs = [("SepalLengthCm", "SepalWidthCm"), ("PetalLengthCm", "PetalWidthCm")]
for ax, (fx, fy) in zip(axes, pairs):
    for cls in classes:
        subset = df[df[TARGET_COLUMN] == cls]
        ax.scatter(subset[fx], subset[fy], label=cls, alpha=0.7)
    ax.set_xlabel(FEATURE_LABELS[fx])
    ax.set_ylabel(FEATURE_LABELS[fy])
    ax.set_title(f"{FEATURE_LABELS[fx][:-5]} vs {FEATURE_LABELS[fy][:-5]}")
    ax.grid(True)
axes[0].legend()
save_and_show("feature_relationships.png")

print(f"\nFigures saved to: {FIGURE_DIR}")
