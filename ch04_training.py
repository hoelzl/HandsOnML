# %%
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn import datasets
# from sklearn.base import clone
from sklearn.linear_model import (LinearRegression, LogisticRegression, SGDRegressor, Ridge, Lasso, ElasticNet)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# %%
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# %%
plt.figure(figsize=(10, 6))
plt.scatter(X, y)
plt.show()

# %%
X_b = np.c_[np.ones((100, 1)), X]  # Add X[i, 0] = 1 for each i
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(f"Best parameters: intercept = {theta_best[0, 0]:.2f}, coef = {theta_best[1, 0]:.2f}")

# %%
np.set_printoptions(precision=1, floatmode="fixed")

# %%
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
print(f"Precise values for {X_new[:, 0]}: {3.0 * X_new[:, 0] + 4.0}")
print(f"Predictions for    {X_new[:, 0]}: {y_predict[:, 0]}")

# %%
np.set_printoptions(precision=None, floatmode=None)

# %%
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

# %%
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(f"Best parameters:   intercept = {theta_best[0, 0]:.2f}, coef = {theta_best[1, 0]:.2f}")
print(f"Linear regression: intercept = {lin_reg.intercept_[0]:.2f}, coef = {lin_reg.coef_[0, 0]:.2f}")

# %%
print(f"Precise values for   {X_new[:, 0]}: {3.0 * X_new[:, 0] + 4.0}")
print(f"Predictions (LR) for {X_new[:, 0]}: {y_predict[:, 0]}")

# %%
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
print(f"Parameters using lstsq(): intercept = {theta_best_svd[0, 0]:.2f}, coef = {theta_best_svd[1, 0]:.2f}")

# %%
theta_best_pinv = np.linalg.pinv(X_b).dot(y)
print(f"Parameters using pinv(): intercept = {theta_best_pinv[0, 0]:.2f}, coef = {theta_best_pinv[1, 0]:.2f}")

# %%
eta = 0.1  # learning rate
n_iterations = 1_000
m = 100
theta = np.random.randn(2, 1)  # random initialization

# %%
for iteration in range(n_iterations):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

# %%
print(f"Parameters using gradient descent: intercept = {theta[0, 0]:.2f}, coef = {theta[1, 0]:.2f}")

# %%
n_epochs = 50
t0, t1 = 5, 50


# %%
def learning_schedule(t):
    return t0 / (t + t1)


# %%
theta = np.random.randn(2, 1)

# %%
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients

# %%
print(f"Parameters using stochastic gradient descent: intercept = {theta[0, 0]:.2f}, coef = {theta[1, 0]:.2f}")

# %%
print("y.shape =", y.shape, "y.ravel().shape =", y.ravel().shape)

# %%
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())

# %%
print(f"Parameters for sgd_reg: intercept = {sgd_reg.intercept_[0]:.2f}, coef = {sgd_reg.coef_[0]:.2f}")

# %%
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# %%
plt.plot(X, y, "b.")
plt.show()

# %%
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
print(f"X[0]: {X[0]}, X_poly[0]: {X_poly[0]}")

# %%
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print("intercept:", lin_reg.intercept_, "coef:", lin_reg.coef_)

# %%
plt.plot(*zip(*sorted(zip(X, lin_reg.predict(X_poly)))), "r-")
plt.plot(X, y, "b.")
plt.show()


# %%
# noinspection PyShadowingNames
def compute_learning_curves(model, X, y, extra_epochs=10):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []

    # noinspection PyShadowingNames
    def compute_learning_curve_value(m, model):
        # if "coef_" in dir(model) and "intercept_" in dir(model):
        #     print(m, model.coef_, model.intercept_)
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    for m in range(1, len(X_train)):
        # compute_learning_curve_value(m, clone(model))
        compute_learning_curve_value(m, model)
    for m in range(extra_epochs):
        compute_learning_curve_value(len(X_train), model)

    return train_errors, val_errors


# %%
# noinspection PyShadowingNames
def plot_learning_curves(model, X, y, max_cap=3, extra_epochs=10):
    train_errors, val_errors = compute_learning_curves(model, X, y, extra_epochs=extra_epochs)
    plt.plot(np.clip(np.sqrt(train_errors), 0, max_cap), "r-+", linewidth=2, label="train")
    plt.plot(np.clip(np.sqrt(val_errors), 0, max_cap), "b-", linewidth=3, label="val")
    plt.show()


# %%
# noinspection PyShadowingNames
def plot_avg_learning_curves(model, epochs, X, y, max_cap=3, extra_epochs=10):
    all_train_errors, all_val_errors = [], []
    for e in range(epochs):
        train_errors, val_errors = compute_learning_curves(model, X, y, extra_epochs=extra_epochs)
        all_train_errors.append(train_errors)
        all_val_errors.append(val_errors)
    avg_train_errors = np.average(all_train_errors, axis=0)
    avg_val_errors = np.average(all_val_errors, axis=0)
    plt.plot(np.clip(np.sqrt(avg_train_errors), 0, max_cap), "r-+", linewidth=2, label="train")
    plt.plot(np.clip(np.sqrt(avg_val_errors), 0, max_cap), "b-", linewidth=3, label="val")
    plt.show()


# %%
lin_reg = LinearRegression(n_jobs=64)
plot_learning_curves(lin_reg, X, y)

# %%
plot_avg_learning_curves(lin_reg, 25, X, y)

# %%
polynomial_regression = Pipeline([
    ("poly_features", PolynomialFeatures(degree=12, include_bias=False)),
    ("lin_reg", LinearRegression(n_jobs=32)),
])
# Increase the number of epochs to something like 200 to get more consistent curves
plot_avg_learning_curves(polynomial_regression, 20, X, y, max_cap=5, extra_epochs=25)

# %%
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])

# %%
sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])

# %%
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])

# %%
sgd_reg = SGDRegressor(penalty="l1")
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])

# %%
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])

# %%
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.2)

# %%
poly_scaler = Pipeline([
    ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
    ("std_scaler", StandardScaler())
])
X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.fit_transform(X_val)

# %%
sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0005)

# %%
minimum_val_error = float("inf")
best_epoch = None
best_model = None

# %%
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train.ravel())
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val, y_val_predict)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = deepcopy(sgd_reg)

# %%
print(f"Best epoch: {best_epoch}, minimum validation error: {minimum_val_error:.2f}")

# %%
iris = datasets.load_iris()
print(list(iris.keys()))

# %%
X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int32)

# %%
log_reg = LogisticRegression(n_jobs=32)
log_reg.fit(X, y)

# %%
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris virginica")
plt.show()

# %%
print("Predictions for 1.7cm and 1.5cm:", log_reg.predict([[1.7], [1.5]]))

# %%
X = iris["data"][:, (2, 3)]
y = iris["target"]

# %%
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X, y)

# %%
print(softmax_reg.predict([[5, 2]]), softmax_reg.predict_proba([[5, 2]]))

# %%

