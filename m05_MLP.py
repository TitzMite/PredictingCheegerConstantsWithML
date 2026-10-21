
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, KFold

import m01_Graphs as Gra

#################################

X       = Gra.X
y       = Gra.y
X_train = Gra.X_train
X_test  = Gra.X_test
y_train = Gra.y_train
y_test  = Gra.y_test
X_gap    = Gra.X[:, 1]
Xg_train = Gra.X_train[:, :1]
Xg_test  = Gra.X_test[:, :1]

###############################

pipeline = make_pipeline(
    StandardScaler(),
    MLPRegressor(random_state=43)
)

parameters = {
    "mlpregressor__hidden_layer_sizes": [(32,), (64,), (64, 32)],
    "mlpregressor__alpha": [0.0001, 0.01, 1.0],
    "mlpregressor__max_iter": [500, 1000]
}

###################################

search = GridSearchCV(
    pipeline,
    param_grid=parameters,
    scoring="neg_mean_absolute_error",
    cv=KFold(n_splits=5, shuffle=True, random_state=43),
    n_jobs=-1,
)

search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)

model = search.best_estimator_

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("We trained an MLP model with the whole spectrum.\n")

Gra.evaluate_model(y_test, y_pred, "MLP, whole spectrum")

##################################

gap_search = GridSearchCV(
    pipeline,
    param_grid=parameters,
    scoring="neg_mean_absolute_error",
    cv=KFold(n_splits=5, shuffle=True, random_state=43),
    n_jobs=-1,
)

gap_search.fit(Xg_train, y_train)

print("Best parameters:", gap_search.best_params_)

gap_model = gap_search.best_estimator_

yg_pred = gap_model.predict(Xg_test)

print("We trained an MLP model with the spectral gap only.\n")

Gra.evaluate_model(y_test, yg_pred, "MLP, spectral gap only")

print("\n\n----------------------------\n\n")