import argparse
import joblib
import json
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

def train(version: str):
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if version == "v0.1":
        regressor = LinearRegression()
    else:
        regressor = RandomForestRegressor(random_state=42, n_estimators=100)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', regressor)
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)

    joblib.dump(pipeline, "model.joblib")
    with open("metrics.json", "w") as f:
        json.dump({"rmse": rmse, "version": version}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True, choices=["v0.1", "v0.2"])
    args = parser.parse_args()
    train(args.version)