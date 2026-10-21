
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

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

n_estimators = 200
random_state = 43

model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
gap_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

###########################

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("We trained a random forest model with the whole spectrum.\n")

Gra.evaluate_model(y_test, y_pred, "Random Forest, whole spectrum")

###########################

gap_model.fit(Xg_train, y_train)

yg_pred = gap_model.predict(Xg_test)

print("We trained a random forest model with only the spectral gap.\n")

Gra.evaluate_model(y_test, yg_pred, "Random Forest, spectral gap only")