import logging
import os
import sys
import warnings
from urllib.parse import urlparse
import tarfile
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from http.client import HTTP_VERSION_NOT_SUPPORTED
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def eda(housing):
    housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
    housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
    housing["population_per_household"]=housing["population"]/housing["households"]
    return housing

def train_test_split_dataset_creation(housing):
    X = housing.drop("median_house_value", axis=1).copy()

    Y = housing['median_house_value']
    x_train,x_test,y_train,y_test = train_test_split(X,Y, stratify = X['ocean_proximity'], test_size = 0.2)
    return  x_train,x_test,y_train,y_test

def pipeline_setup(x_train,x_test):

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
    housing_num = x_train.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num.columns)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])
    x_train = full_pipeline.fit_transform(x_train)
    x_test = full_pipeline.transform(x_test)
    return x_train, x_test

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    # Downloading Data
    fetch_housing_data()
    # Reading Data
    housing = load_housing_data()
    # Basic EDA
    housing = eda(housing)
    x_train,x_test,y_train,y_test = train_test_split_dataset_creation(housing)
    x_train, x_test = pipeline_setup(x_train,x_test)

    print(x_test.shape,x_train.shape)

    # alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    # l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}
        ]

        forest_reg = RandomForestRegressor()

        grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                            scoring='neg_mean_squared_error',
                            return_train_score=True)

        grid_search.fit(x_train, y_train)
        tuned_model = grid_search.best_estimator_
        params = grid_search.best_params_
        print(tuned_model)

        predicted_qualities = tuned_model.predict(x_test)

        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("n_estimators", params['n_estimators'])
        mlflow.log_param("max_features", params['max_features'])
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        mlflow.sklearn.log_model(tuned_model, "model")
