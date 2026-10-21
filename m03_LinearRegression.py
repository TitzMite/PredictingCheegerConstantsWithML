
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

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

##################

model = make_pipeline(StandardScaler(), LinearRegression())

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("We trained a linear regression model with the whole spectrum.\n")

Gra.evaluate_model(y_test, y_pred, "Linear regression, whole spectrum")

########################

print("\n------------------------------\n")

########################

gap_model = make_pipeline(StandardScaler(), LinearRegression())

gap_model.fit(Xg_train, y_train)

yg_pred = gap_model.predict(Xg_test)

print("We trained a linear regression model with only the spectral gap. \n")

Gra.evaluate_model(y_test, y_pred, "Linear regression, spectral gap only")