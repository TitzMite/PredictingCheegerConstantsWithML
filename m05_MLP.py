
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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

model = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(64, 32),max_iter=500,random_state=43)
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("We trained an MLP model with the whole spectrum.\n")

Gra.evaluate_model(y_test, y_pred, "MLP, whole spectrum")

##################################

model = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(64, 32),max_iter=500,random_state=43)
)

model.fit(Xg_train, y_train)

yg_pred = model.predict(Xg_test)

print("We trained an MLP model with the spectral gap only.\n")

Gra.evaluate_model(y_test, y_pred, "MLP, spectral gap only")