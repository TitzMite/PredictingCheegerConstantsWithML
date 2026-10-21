
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
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

###############

n_estimators = 200
random_state = 43
###########################

parameters = {
    "n_estimators": [50, 100],
    "max_depth": [None, 15],
    "min_samples_leaf": [1, 5]
}

########################

search = GridSearchCV(
    RandomForestRegressor(random_state=43),
    param_grid=parameters,
    scoring="neg_mean_absolute_error",
    cv=KFold(n_splits=5, shuffle=True, random_state=43),
    n_jobs=-1,
)

print("searching good parameters for a random forest model. \n")

search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)

model = search.best_estimator_

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("We trained a random forest model with the whole spectrum.\n")

Gra.evaluate_model(y_test, y_pred, "Random Forest, whole spectrum")

###########################

search_gap = GridSearchCV(
    RandomForestRegressor(random_state=43),
    param_grid=parameters,
    scoring="neg_mean_absolute_error",
    cv=KFold(n_splits=5, shuffle=True, random_state=43),
    n_jobs=-1,
)

print("searching good parameters for a random forest model. \n")

search_gap.fit(Xg_train, y_train)

print("Best parameters:", search_gap.best_params_)

gap_model = search.best_estimator_

gap_model.fit(Xg_train, y_train)

yg_pred = gap_model.predict(Xg_test)

print("We trained a random forest model with only the spectral gap.\n")

Gra.evaluate_model(y_test, yg_pred, "Random Forest, spectral gap only")

print("\n\n----------------------------\n\n")
