# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml, make_swiss_roll
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import LocallyLinearEmbedding, TSNE
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

# %%
np.random.seed(4)

# %%
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

# %%
angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles) / 2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

# %%
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]
print(f"Shapes: U: {U.shape}, s: {s.shape}, Vt: {Vt.shape}")

# %%
m, n = X_centered.shape

S = np.zeros(X_centered.shape)
S[:n, :n] = np.diag(s)

# %%
print(f"X_centered = U @ S @ Vt? {np.allclose(X_centered, U @ S @ Vt)}")

# %%
W2 = Vt.T[:, :2]
X2D_v1 = X_centered.dot(W2)

# %%
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
print(f"X2d == X2d_v1? {np.allclose(np.abs(X2D), np.abs(X2D_v1))}")
# print(f"Components:\n{pca.components_.T}")
# print(f"First component: {pca.components_.T[:, 0]}")
print(f"First component: {pca.components_[0]}")
print(f"Explained variance: {pca.explained_variance_}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# %%
X_train, X_val = train_test_split(X, test_size=0.2)

# %%
pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1

# %%
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)
print(f"Explained variance ratio: {pca.explained_variance_ratio_} "
      f"({np.sum(pca.explained_variance_ratio_):.3f})")

# %%
if "mnist" in globals():
    mnist = globals()["mnist"]
else:
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    mnist.target = mnist.target.astype(np.uint8)

# %%
X = mnist["data"]
y = mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

# %%
pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1

# %%
plt.figure(figsize=(6, 4), dpi=300)
plt.plot(cumsum, linewidth=3)
plt.axis([0, 400, 0, 1])
plt.xlabel("Dimensions")
plt.ylabel("Explained Variance")
plt.plot([d, d], [0, 0.95], "k:")
plt.plot([0, d], [0.95, 0.95], "k:")
plt.plot(d, 0.95, "ko")
plt.grid(True)
plt.show()

# %%
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)

# %%
print(f"Components: {pca.components_.shape[0]}")


# %%
def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    n_rows = (len(instances) - 1) // images_per_row + 1
    n_empty = n_rows * images_per_row - len(instances)
    padded_instances = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0)
    image_grid = padded_instances.reshape((n_rows, images_per_row, size, size))
    big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size, images_per_row * size)
    plt.imshow(big_image, cmap="binary", **options)
    plt.axis("off")


# %%
# noinspection PyShadowingNames
def plot_recovered(X_recovered):
    plt.figure(figsize=(7, 4), dpi=300)
    plt.subplot(121)
    plot_digits(X_train[::2100])
    plt.title("Original")
    plt.subplot(122)
    plot_digits(X_recovered[::2100])
    plt.title("Compressed")


# %%
plot_recovered(X_recovered)
plt.show()

# %%
pca = PCA(n_components=32)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)

# %%
plot_recovered(X_recovered)
plt.show()

# %%
rnd_pca = PCA(n_components=154, svd_solver="randomized")
X_reduced = rnd_pca.fit_transform(X_train)
X_recovered = rnd_pca.inverse_transform(X_reduced)

# %%
plot_recovered(X_recovered)
plt.show()

# %%
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)

# %%
for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)

# %%
X_reduced = inc_pca.transform(X_train)
X_recovered = inc_pca.inverse_transform(X_reduced)

# %%
plot_recovered(X_recovered)
plt.show()

# %%
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
y = t > 6.9

# %%
rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)

# %%
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap="hot")
plt.show()

# %%
clf = Pipeline([
    ("kpca", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression()),
])

# %%
param_grid = [{
    "kpca__gamma": np.linspace(0.03, 0.05, 10),
    "kpca__kernel": ["rbf", "sigmoid"]
}]

# %%
grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)

# %%
print(f"Best parameters: {grid_search.best_params_}")

# %%
rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)

# %%
print(f"MSE for Kernel PCA: {mean_squared_error(X, X_preimage):.1f}")

# %%
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_reduced = lle.fit_transform(X)

# %%
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap="hot")
plt.show()

# %%
tsne = TSNE(n_components=2, learning_rate="auto", init="random", n_jobs=48, random_state=42)
X_reduced = tsne.fit_transform(X)

# %%
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap="hot")
plt.show()
