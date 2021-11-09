# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR

# %%
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]
y = (iris["target"] == 2).astype(np.float64)

# %%
svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge"))
])
svm_clf.fit(X, y)

# %%
print(svm_clf.predict([[5.5, 1.7]]))

# %%
X, y = make_moons(n_samples=100, noise=0.15)
polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss="hinge")),
])
polynomial_svm_clf.fit(X, y)

# %%
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# %%
plt.scatter(X[:, 0], X[:, 1], c=polynomial_svm_clf.predict(X))
plt.show()

# %%
plt.scatter(X[:, 0], X[:, 1], c=(y == polynomial_svm_clf.predict(X)))
plt.show()

# %%
poly_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5)),
])
poly_kernel_svm_clf.fit(X, y)

# %%
plt.scatter(X[:, 0], X[:, 1], c=(y == poly_kernel_svm_clf.predict(X)))
plt.show()

# %%
rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
])
rbf_kernel_svm_clf.fit(X, y)

# %%
plt.scatter(X[:, 0], X[:, 1], c=(y == rbf_kernel_svm_clf.predict(X)))
plt.show()

# %%
rbf_kernel_svm_clf_2 = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=0.1, C=1000))
])
rbf_kernel_svm_clf_2.fit(X, y)

# %%
plt.scatter(X[:, 0], X[:, 1], c=(y == rbf_kernel_svm_clf_2.predict(X)))
plt.show()

# %%
svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y)

# %%
svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X, y)

# %%
