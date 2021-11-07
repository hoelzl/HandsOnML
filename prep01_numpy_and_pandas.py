# %%
import numpy as np
import pandas as pd

# %% [md]
# # Pandas DataFrames: Pivot
#
# The `pivot()` method reorganizes a table based on its values.
# Builds a new data frame using the unique values of one or more indices or columns.

# %%
df = pd.DataFrame({
    "col_1": ["1_1"] * 4 + ["1_2"] * 4 + ["1_3"] * 4,
    "col_2": ["2_1"] * 3 + ["2_2"] * 3 + ["2_3"] * 3 + ["2_4"] * 3,
    "col_3": "3_1 3_1 3_2 3_2 3_3 3_3 3_4 3_4 3_5 3_5 3_6 3_6".split(),
    "col_4": "4_1 4_2 4_3 4_4 4_5 4_6 4_7 4_8 4_9 4_A 4_B 4_C".split(),
})

# %%
pivot_1 = df.pivot(columns="col_1", values="col_2")
pivot_2 = df.pivot(columns="col_1", values="col_3")
pivot_3 = df.pivot(index="col_4", columns="col_1", values="col_2")
pivot_4 = df.pivot(columns="col_4", values="col_1")

# %% [md]
# # Stacking vectors/arrays using numpy.
#

# %%
vec_1 = np.array([11, 12, 13, 14])
vec_2 = np.array([21, 22, 23, 24])
vec_3 = np.array([31, 32, 33, 34])

# %%
vec_c2 = np.c_[vec_1, vec_2]
vec_c3 = np.c_[vec_1, vec_2, vec_3]

# %%
mat_1 = np.array([[111, 112, 113], [121, 122, 123]])
mat_2 = np.array([[211, 212, 213], [221, 222, 223]])
mat_3 = np.array([[311, 312, 313], [321, 322, 323]])

# %%
vec_c22 = np.c_[mat_1, mat_2]
vec_c23 = np.c_[mat_1, mat_2, mat_3]

# %%
mat_4 = np.array([[111, 112], [121, 122], [131, 132], [141, 142]])
vec_c_mixed = np.c_[vec_1, mat_4, vec_2]

# %%

