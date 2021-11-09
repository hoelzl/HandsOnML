# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
import xgboost
from pathlib import Path
from sklearn.datasets import make_moons, load_iris
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier, BaggingClassifier,
                              ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# %%
np.random.seed(42)
IMAGES_PATH = Path("data/images")

# %%
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# %%
log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(gamma="scale", random_state=42)

# %%
voting_clf = VotingClassifier(
    estimators=(("lr", log_clf), ("rf", rnd_clf), ("svc", svm_clf)),
    voting="hard")
voting_clf.fit(X_train, y_train)


# %%
def evaluate_ensemble(classifiers=(log_clf, rnd_clf, svm_clf, voting_clf)):
    for clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # noinspection SpellCheckingInspection
        print(f"{clf.__class__.__name__ + ':':<24}"
              f"Acc: {accuracy_score(y_test, y_pred):.3f}, "
              f"Prec: {precision_score(y_test, y_pred):.3f}, "
              f"Rec: {recall_score(y_test, y_pred):.3f}, "
              f"F1: {f1_score(y_test, y_pred):.3f}")


# %%
evaluate_ensemble()

# %%
svm_clf = SVC(gamma="scale", probability=True, random_state=42)
voting_clf = VotingClassifier(
    estimators=(("lr", log_clf), ("rf", rnd_clf), ("svc", svm_clf)),
    voting="soft")
evaluate_ensemble()

# %%
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=48,
)
bag_clf.fit(X_train, y_train)
evaluate_ensemble((bag_clf,))

# %%
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=48, oob_score=True,
)
bag_clf.fit(X_train, y_train)
evaluate_ensemble((bag_clf,))
print(f"{'OOB score:':<24}Oob: {bag_clf.oob_score_:.3f}")

# %%
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=48)
rnd_clf.fit(X_train, y_train)
evaluate_ensemble((bag_clf,))

# %%
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_features="auto", max_leaf_nodes=16),
    n_estimators=500, max_samples=100, bootstrap=True, n_jobs=48,
)
bag_clf.fit(X_train, y_train)
evaluate_ensemble((bag_clf,))

# %%
ext_clf = ExtraTreesClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=48)
ext_clf.fit(X_train, y_train)
evaluate_ensemble((ext_clf,))

# %%
iris = load_iris()

# %%
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=48)
rnd_clf.fit(iris["data"], iris["target"])
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(f"{name + ':':<18} {score:.3f}")

# %%
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=100,
    algorithm="SAMME.R", learning_rate=0.5,
)
ada_clf.fit(X_train, y_train)
evaluate_ensemble([ada_clf])

# %%
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)

# %%
tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)

# %%
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg2.fit(X, y2)

# %%
y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg3.fit(X, y3)

# %%
X_new = np.array([[0.1], [0.4], [0.8]])
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
print("True value:                    ", 3 * X_new.reshape(-1) ** 2 + 0.05)
print("Prediction (Gradient Boosting):", y_pred)

# %%
X_plot = np.linspace(-0.6, 0.6, 100).reshape(-1, 1)
plt.plot(X_plot, sum(tree.predict(X_plot) for tree in (tree_reg1, tree_reg2, tree_reg3)))
plt.plot(X_plot, 3 * X_plot.reshape(-1) ** 2 + 0.05)
plt.plot(X, y, ".")
plt.show()

# %%
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(X, y)

# %%
X = np.random.rand(100, 1) - 0.5
y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)
X_train, X_val, y_train, y_val = train_test_split(X, y)

# %%
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(X_train, y_train)

# %%
errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors) + 1

# %%
gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators)
gbrt_best.fit(X_train, y_train)

# %%
min_error = np.min(errors)


# %%
# noinspection PyShadowingNames
def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.",
                     data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)


# %%
plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.plot(errors, "b.-")
plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], "k--")
plt.plot([0, 120], [min_error, min_error], "k--")
plt.plot(bst_n_estimators, min_error, "ko")
plt.text(bst_n_estimators, min_error * 1.2, "Minimum", ha="left", fontsize=14)
plt.axis([0, 120, 0, 0.01])
plt.xlabel("Number of trees")
plt.ylabel("Error")
plt.title("Validation error")

plt.subplot(122)
plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
plt.title(f"Best model ({bst_n_estimators} trees)")
plt.ylabel("$y$", rotation=0)
plt.xlabel("$x_1")

plt.show()

# %%
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=2)

# %%
min_val_error = float("inf")
error_going_up = 0

# %%
for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break

# %%
xgb_reg = xgboost.XGBRegressor()
xgb_reg.fit(X_train, y_train)
y_pred = xgb_reg.predict(X_val)
print(f"RMSE for XGB: {np.sqrt(mean_squared_error(y_val, y_pred)):.4}")

# %%
xgb_reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=2)
y_pred = xgb_reg.predict(X_val)
print(f"RMSE for XGB: {np.sqrt(mean_squared_error(y_val, y_pred)):.4}")

# %%
