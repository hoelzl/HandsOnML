# %%
import numpy as np
import matplotlib.pyplot as plt

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
print(f"Best parameters: bias = {theta_best[0, 0]:.2f}, weight = {theta_best[1, 0]:.2f}")

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

