# %%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# %%
data_path = Path("data/satisfaction")

# %%
oecd_bli_df = pd.read_csv(data_path / "oecd_bli_2015.csv", thousands=",", encoding="utf-8")

# %%
gdp_per_capita_df = pd.read_csv(data_path / "gdp_per_capita.csv", thousands=",", delimiter="\t",
                                na_values="n/a", encoding="utf-8")


# %%
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


# %%
country_stats = prepare_country_stats(oecd_bli_df, gdp_per_capita_df)

# %%
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# %%
country_stats.plot(kind="scatter", x="GDP per capita", y="Life satisfaction")
plt.show()

# %%
linear_model = LinearRegression()

# %%
linear_model.fit(X, y)

# %%
X_cyprus = [[22587]]
print(linear_model.predict(X_cyprus))

# %%
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X, y)

# %%
print(knn_model.predict(X_cyprus))

