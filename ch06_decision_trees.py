# %%
import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz

# %%
image_path = Path("data/images")
image_path.mkdir(exist_ok=True, parents=True)

# %%
iris = load_iris()
X = iris.data[:, 2:]
y = iris.target

# %%
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

# %%
export_graphviz(tree_clf, out_file=(image_path / "iris_tree.dot").as_posix(),
                feature_names=iris.feature_names[2:], class_names=iris.target_names,
                rounded=True, filled=True)

# %%
np.set_printoptions(precision=3)
print("Class probabilities:", tree_clf.predict_proba([[5, 1.5]]))
print("Predicted class:", tree_clf.predict([[5, 1.5]]))

# %%
np.random.seed(42)
m = 200
X = np.random.rand(m, 1)
y = 4 * (X - 0.5) ** 2
y = y + np.random.randn(m, 1) / 10

# %%
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)

# %%
export_graphviz(tree_reg, out_file=(image_path / "iris_tree_reg.dot").as_posix(),
                rounded=True, filled=True)

# %%
tree_reg1 = DecisionTreeRegressor(random_state=42, max_depth=2)
tree_reg2 = DecisionTreeRegressor(random_state=42, max_depth=3)
tree_reg1.fit(X, y)
tree_reg2.fit(X, y)


# %%
# noinspection PyShadowingNames
def plot_regression_predictions(tree_reg, X, y, axes_min_max=(0, 1, -0.2, 1), ylabel="y"):
    x1 = np.linspace(axes_min_max[0], axes_min_max[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes_min_max)
    plt.xlabel("X")
    if ylabel:
        plt.ylabel(ylabel, rotation=0)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred, "r.-", linewidth=2)


# %%
# noinspection PyTypeChecker
fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
# Set current axes to axes[0]
plt.sca(axes[0])
plot_regression_predictions(tree_reg1, X, y)
plt.sca(axes[1])
plot_regression_predictions(tree_reg2, X, y)
plt.show()
