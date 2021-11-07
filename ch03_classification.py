# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone, BaseEstimator
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score, fbeta_score,
                             precision_recall_curve, ConfusionMatrixDisplay, roc_curve, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
# Needed for the commented-out part
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import SVC

# %%
# Don't reload the dataset when re-evaluating the file.
if "mnist" in globals():
    # To avoid warnings about undefined variables
    mnist = globals()["mnist"]
else:
    mnist = fetch_openml("mnist_784", version=1)

# %%
mnist.keys()

# %%
X: pd.DataFrame = mnist["data"]
y: pd.DataFrame = mnist["target"]

# %%
print(X.shape, y.shape, flush=True)

# %%
some_digit: pd.DataFrame = X.iloc[[0]]
some_digit_image = some_digit.to_numpy().reshape(28, 28)

# %%
plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.title(f"Value: {y[0]}")
plt.show()

# %%
y = y.astype(np.uint8)

# %%
X_train, X_test, y_train, y_test = X.iloc[:60_000], X.iloc[60_000:], y.iloc[:60_000], y.iloc[60_000:]

# %%
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# %%
sgd_clf = SGDClassifier(random_state=42, n_jobs=32)
sgd_clf.fit(X_train, y_train_5)

# %%
sgd_clf.predict(some_digit)

# %%
sk_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# %%
for train_index, test_index in sk_folds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    x_train_folds: pd.DataFrame = X_train.iloc[train_index]
    y_train_folds: pd.Series = y_train_5.iloc[train_index]
    x_test_fold: pd.DataFrame = X_train.iloc[test_index]
    y_test_fold: pd.Series = y_train_5.iloc[test_index]

    clone_clf.fit(x_train_folds, y_train_folds)
    y_pred = clone_clf.predict(x_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(f"Fraction correct: {n_correct / len(y_pred)}")

# %%
print(f"Cross validation score: {cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy', n_jobs=32)}",
      flush=True)


# %%
# noinspection PyUnusedLocal,PyShadowingNames
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        return self

    # noinspection PyMethodMayBeStatic
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


# %%
never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# %%
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, n_jobs=32)

# %%
sgd_confusion_matrix = confusion_matrix(y_train_5, y_train_pred)
print(sgd_confusion_matrix)

# %%
ConfusionMatrixDisplay.from_predictions(y_train_5, y_train_pred)
plt.show()

# %%
ConfusionMatrixDisplay.from_predictions(y_train_5, y_train_pred, normalize="true")
plt.show()

# %%
ConfusionMatrixDisplay.from_predictions(y_train_5, y_train_pred, normalize="pred")
plt.show()

# %%
ConfusionMatrixDisplay.from_predictions(y_train_5, y_train_pred, normalize="all")
plt.show()

# %%
y_train_perfect_predictions = y_train_5
confusion_matrix(y_train_5, y_train_perfect_predictions)

# %%
ConfusionMatrixDisplay.from_predictions(y_train_5, y_train_perfect_predictions, normalize="true")
plt.show()

# %%
ConfusionMatrixDisplay.from_predictions(y_train_5, y_train_perfect_predictions, normalize="all")
plt.show()

# %%
print(f"precision: {precision_score(y_train_5, y_train_pred)}")
print(f"recall:    {recall_score(y_train_5, y_train_pred)}")
print(f"f1:        {f1_score(y_train_5, y_train_pred)}", flush=True)

# %%
print(f"f_beta(β=1):   {fbeta_score(y_train_5, y_train_pred, beta=1.0)}")
print(f"f_beta(β=0.5): {fbeta_score(y_train_5, y_train_pred, beta=0.5)}")
print(f"f_beta(β=2):   {fbeta_score(y_train_5, y_train_pred, beta=2.0)}")

# %%
y_scores = sgd_clf.decision_function(some_digit)
print(y_scores)

# %%
threshold = 0
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

# %%
threshold = 10_000
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

# %%
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function", n_jobs=32)

# %%
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


# %%
# noinspection PyShadowingNames
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="recall")


# %%
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

# %%
plt.plot(recalls, precisions)
plt.show()

# %%
threshold_90_precision = thresholds[np.argmax(precisions >= 0.9)]
print(threshold_90_precision)

# %%
y_train_pred_90 = (y_scores >= threshold_90_precision)

# %%
print(f"Precision: {precision_score(y_train_5, y_train_pred_90)}")
print(f"Recall:    {recall_score(y_train_5, y_train_pred_90)}")

# %%
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


# %%
# noinspection PyShadowingNames
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")


# %%
plot_roc_curve(fpr, tpr)
plt.show()

# %%
print(f"ROC AUC score: {roc_auc_score(y_train_5, y_scores):.3f}")

# %%
forest_clf = RandomForestClassifier(random_state=42, n_jobs=32)
y_probs_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba", n_jobs=32)
y_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, n_jobs=32)

# %%
print([bool(np.argmax(pred)) for pred in y_probs_forest[:10]])
print(list(y_pred_forest[:10]))

# %%
y_pred_05_forest = np.array([bool(np.argmax(pred)) for pred in y_probs_forest])
assert all(y_pred_forest == y_pred_05_forest), "predictions from probability not equal to predictions?"

# %%
y_scores_forest = y_probs_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

# %%
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()

# %%
print(f"ROC AUC score (RF): {roc_auc_score(y_train_5, y_scores_forest):.3f}")
print(f"Precision (RF):     {precision_score(y_train_5, y_pred_forest):.3f}")
print(f"Recall (RF):        {recall_score(y_train_5, y_pred_forest):.3f}")

# Commented out since it's pretty slow.
# # %%
# svm_clf: SVC = SVC()
# svm_clf.fit(x_train, y_train)
# print(svm_clf.predict(some_digit), flush=True)
#
# # %%
# some_digit_scores = svm_clf.decision_function(some_digit)
# print(some_digit_scores)
# print("Index of highest score:", np.argmax(some_digit_scores))
# print("Classifier classes:    ", svm_clf.classes_)
# print("Classifier prediction: ", svm_clf.classes_[np.argmax(some_digit_scores)])
#
# # %%
# ovr_clf = OneVsRestClassifier(SVC(), n_jobs=32)
# ovr_clf.fit(x_train, y_train)
# print("OVR prediction:", ovr_clf.predict(some_digit))

# %%
sgd_clf = SGDClassifier(n_jobs=32)
sgd_clf.fit(X_train, y_train)
print("SGD prediction:", sgd_clf.predict(some_digit))
print("SGD decision function:", sgd_clf.decision_function(some_digit))

# %%
print("SGD cross val:", cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy", n_jobs=32))

# %%
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
print("SCD CV scaled:", cross_val_score(sgd_clf, x_train_scaled, y_train, cv=3, scoring="accuracy", n_jobs=32))

# %%
y_train_pred = cross_val_predict(sgd_clf, x_train_scaled, y_train, cv=3, n_jobs=32)

# %%
conf_mx = confusion_matrix(y_train, y_train_pred)
print(conf_mx)

# %%
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

# %%
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
plt.show()

# %%
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

# %%
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap="coolwarm")
plt.show()

# %%
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

# %%
knn_clf = KNeighborsClassifier(n_jobs=32)
knn_clf.fit(X_train, y_multilabel)

# %%
knn_clf.predict(some_digit)

# %%
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3, n_jobs=32)
print("F1 score:", f1_score(y_multilabel, y_train_knn_pred, average="macro"))

# %%
noise = np.random.randint(0, 100, (len(X_train), 28 * 28))
x_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 28 * 28))
x_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

# %%
knn_clf.fit(x_train_mod, y_train_mod)

# %%
some_index = 1
plt.imshow(x_test_mod.iloc[[some_index]].to_numpy().reshape(28, 28), cmap="binary")
plt.show()

# %%
plt.imshow(y_test_mod.iloc[[some_index]].to_numpy().reshape(28, 28), cmap="binary")
plt.show()


# %%
clean_digit = knn_clf.predict(x_test_mod.iloc[[some_index]]).reshape(28, 28)
plt.imshow(clean_digit, cmap="binary")
plt.show()

