
import numpy as np

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

Gra.evaluate_model(y_test, yg_pred, "Linear regression, spectral gap only")

print("\n\n----------------------------\n\n")

############################################

def print_lr_analysis(pipeline, name, top_n=10):
    scaler = pipeline.named_steps["standardscaler"]
    lr = pipeline.named_steps["linearregression"]

    weights = lr.coef_
    raw_weights = weights / scaler.scale_
    intercept = lr.intercept_ - raw_weights @ scaler.mean_
    order = np.argsort(np.abs(weights))[::-1][:top_n]

    print(f"\n{name}")
    print(f"Number of features: {len(weights)}")
    print(f"Raw intercept:      {intercept:.6f}")
    print("\nCoefficients ranked by absolute standardized weight")
    print(f"{'Eigenvalue':<14}{'Standardized':>16}{'Raw':>14}")
    print("-" * 44)

    for j in order:
        print(
            f"{f'lambda_{j + 2}':<14}"
            f"{weights[j]:>16.6f}"
            f"{raw_weights[j]:>14.6f}"
        )


print_lr_analysis(model, "Full-spectrum LR")

print("\n\n----------------------------\n\n")

print_lr_analysis(gap_model, "Gap-only LR")

print("\n\n----------------------------\n\n")