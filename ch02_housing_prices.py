# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tarfile
import urllib.request

from pandas.plotting import scatter_matrix
from pathlib import Path
from scipy import stats
from scipy.sparse.csr import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from zlib import crc32

# %%
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
HOUSING_PATH = Path("data/housing")
HOUSING_CSV = HOUSING_PATH / "housing.csv"


# %%
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    HOUSING_PATH.mkdir(exist_ok=True, parents=True)
    tgz_path = HOUSING_PATH / "housing.tgz"
    urllib.request.urlretrieve(housing_url, tgz_path)
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)


# %%
if not HOUSING_CSV.exists():
    fetch_housing_data()
assert HOUSING_CSV.exists()


# %%
def load_housing_data(housing_csv=HOUSING_CSV):
    return pd.read_csv(housing_csv)


# %%
housing = load_housing_data()

# %%
housing.info()

# %%
housing["ocean_proximity"].value_counts()

# %%
housing[["housing_median_age", "total_bedrooms"]].describe()

# %%
housing.hist(bins=50, figsize=(20, 15))
plt.show()


# %%
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# %%
train_set, test_set = split_train_test(housing, 0.2)
len(train_set), len(test_set)


# %%
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0x_ffff_ffff < test_ratio * 2 ** 32


# %%
def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# %%
housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
len(train_set), len(test_set)

# %%
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
len(train_set), len(test_set)

# %%
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
len(train_set), len(test_set)

# %%
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],
                               labels=[1, 2, 3, 4, 5])

# %%
housing["income_cat"].hist()
plt.title("Income (Whole Dataset)")
plt.show()

# %%
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
strat_train_set, strat_test_set = None, None
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
len(strat_train_set), len(strat_test_set)

# %%
strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, random_state=42,
                                                   stratify=housing["income_cat"])
len(strat_train_set), len(strat_test_set)

# %%
strat_train_set["income_cat"].hist()
plt.title("Income (strat_train_set)")
plt.show()
strat_test_set["income_cat"].hist()
plt.title("Income (strat_test_set)")
plt.show()

# %%
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
train_set["income_cat"].hist()
plt.title("Income (train_set)")
plt.show()
test_set["income_cat"].hist()
plt.title("Income (test_set)")
plt.show()

# %%
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# %%
housing = strat_train_set.copy()

# %%
housing.plot(kind="scatter", x="longitude", y="latitude")
plt.show()

# %%
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()

# %%
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"] / 40,
             label="population", figsize=(15, 10), c="median_house_value", cmap="jet",
             colorbar=True)
plt.show()

# %%
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# %%
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(15, 10))
plt.show()

# %%
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.show()

# %%
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]
housing["bedrooms_per_household"] = housing["total_bedrooms"] / housing["households"]

# %%
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# %%
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# %%
imputer: SimpleImputer = SimpleImputer(strategy="median")

# %%
housing_num: pd.DataFrame = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

# %%
print(imputer.statistics_)

# %%
print(housing_num.median().values)

# %%
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

# %%
housing_cat = housing[["ocean_proximity"]]

# %%
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

# %%
print(ordinal_encoder.feature_names_in_)
print(ordinal_encoder.categories_)

# %%
cat_encoder = OneHotEncoder()
housing_cat_1hot: csr_matrix = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot[:5])

# %%
housing_cat_1hot.toarray()

# %%
print(cat_encoder.feature_names_in_)
print(cat_encoder.categories_)

# %%
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


# %%
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    # noinspection PyPep8Naming,PyUnusedLocal,PyShadowingNames
    def fit(self, X, y=None):
        return self

    # noinspection PyPep8Naming,PyShadowingNames
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# %%
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# %%
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("attribs_adder", CombinedAttributesAdder()),
    ("std_scaler", StandardScaler()),
])

# %%
housing_num_tr = num_pipeline.fit(housing_num)

# %%
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

# %%
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

# %%
housing_prepared = full_pipeline.fit_transform(housing)

# %%
print(f"Number of input features: {len(full_pipeline.feature_names_in_)}")
print(full_pipeline.feature_names_in_)
# This does not work, unfortunately
# print(full_pipeline.get_feature_names())
# print(full_pipeline.get_feature_names_out())

# %%
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# %%
some_data = housing.iloc[:5]
some_labels = housing_labels[:5]
some_data_prepared = full_pipeline.transform(some_data)
some_data_predictions = lin_reg.predict(some_data_prepared)
print(f"Predictions: {[round(x, 1) for x in some_data_predictions]}")
print(f"Labels:      {list(some_labels)}")
print(f"Ratios:      {[round(pred / label, 2) for pred, label in zip(some_data_predictions, some_labels)]}")

# %%
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(f"RMSE: {lin_rmse:.1f}")

# %%
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

# %%
# noinspection DuplicatedCode
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(f"RMSE: {tree_rmse:.1f}")

# %%
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# %%
# noinspection PyShadowingNames
def display_scores(scores):
    print(f"Scores:\n{scores}")
    print(f"Mean:               {scores.mean():.0f}")
    print(f"Standard deviation: {scores.std():.0f}")


# %%
display_scores(tree_rmse_scores)

# %%
forest_reg = RandomForestRegressor(n_jobs=32)
print("Fitting random forest...", flush=True, end="")
# noinspection DuplicatedCode
forest_reg.fit(housing_prepared, housing_labels)
print("done.", flush=True)

# %%
# noinspection DuplicatedCode
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print(f"RMSE: {forest_rmse:.1f}")

# %%
scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)

# %%
display_scores(forest_rmse_scores)

# %%
param_grid = [
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]}
]

# %%
forest_reg = RandomForestRegressor(n_estimators=32)

# %%
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)

# %%
grid_search.fit(housing_prepared, housing_labels)

# %%
print(grid_search.best_params_)

# %%
best_grid_search_estimator = grid_search.best_estimator_

# %%
cv_res = grid_search.cv_results_

# %%
for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
    print(f"RMSE {np.sqrt(-mean_score):.1f} for {params}")

# %%
feature_importances = [round(x * 100, 1) for x in grid_search.best_estimator_.feature_importances_]
print(f"Number of features: {len(feature_importances)}.")
print(feature_importances)

# %%
extra_attribs = ["rooms_per_household", "pop_per_household", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

# %%
final_model = grid_search.best_estimator_

# %%
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

# %%
X_test_prepared = full_pipeline.transform(X_test)

# %%
final_predictions = final_model.predict(X_test_prepared)

# %%
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(f"RMSE = {final_rmse:.1f}")

# %%
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
confidence_interval = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                                               loc=squared_errors.mean(), scale=stats.sem(squared_errors)))
print("Confidence interval:", confidence_interval)

# %%
svm_reg = RandomForestRegressor(n_jobs=32)
# noinspection DuplicatedCode
svm_reg.fit(housing_prepared, housing_labels)

# %%
# noinspection DuplicatedCode
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
print(f"RMSE: {svm_rmse:.1f}")

# %%
scores = cross_val_score(svm_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
svm_rmse_scores = np.sqrt(-scores)

# %%
display_scores(svm_rmse_scores)

# %%
