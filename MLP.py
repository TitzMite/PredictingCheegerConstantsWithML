import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import Graphs as Gra

eigenvals72 = Gra.eigenvals72
cheegers72 = Gra.cheegers72

X = eigenvals72
X_gap = X[:, [0]]
y = np.array(cheegers72, dtype=float)

indices = np.arange(len(y))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=36)

X_train = X[train_idx]
X_test = X[test_idx]
Xg_train = X_gap[train_idx]
Xg_test = X_gap[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(max_iter=5000, random_state=36))
])

param_grid = [
    {
        "mlp__hidden_layer_sizes": [(8,), (16,), (32,), (32,16)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__solver": ["lbfgs"],
        "mlp__alpha": [1e-4, 1e-3, 1e-2]
    },
    {
        "mlp__hidden_layer_sizes": [(16,), (32,), (32,16)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__solver": ["adam"],
        "mlp__alpha": [1e-4, 1e-3, 1e-2],
        "mlp__learning_rate_init": [0.0003, 0.001],
        "mlp__early_stopping": [True],
        "mlp__n_iter_no_change": [50]
    }
]

def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    errors = np.abs(y_test - y_pred)
    quantiles = [0.5, 0.9, 0.95, 0.99]
    print("\n" + name)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R²:", r2)
    print("Error quantiles:")
    for q, value in zip(quantiles, np.quantile(errors, quantiles)):
        print(f"{int(q * 100)}%: {value:.6f}")
    return y_pred

print("Tuning full spectrum...")
search = GridSearchCV(pipeline, param_grid, scoring="neg_mean_absolute_error", cv=3, n_jobs=-1)
search.fit(X_train, y_train)

print("\nBest full-spectrum parameters:")
print(search.best_params_)
print("Cross-validation MAE:", -search.best_score_)

best_model = search.best_estimator_
y_pred = evaluate("Full spectrum:", best_model, X_test, y_test)

print("\nTuning spectral gap...")
search_gap = GridSearchCV(pipeline, param_grid, scoring="neg_mean_absolute_error", cv=3, n_jobs=-1)
search_gap.fit(Xg_train, y_train)

print("\nBest spectral-gap parameters:")
print(search_gap.best_params_)
print("Cross-validation MAE:", -search_gap.best_score_)

best_gap_model = search_gap.best_estimator_
yg_pred = evaluate("Spectral gap:", best_gap_model, Xg_test, y_test)

baseline_pred = np.full_like(y_test, y_train.mean())
print("\nBaseline MAE:", mean_absolute_error(y_test, baseline_pred))