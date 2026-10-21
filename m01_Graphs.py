
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

###############################

with open("data_graphs84.pkl", "rb") as f:
    graphs84 = pickle.load(f)

eigenvals84 = np.load("data_eigenvals84.npy")

with open("data_cheegers84.pkl", "rb") as f:
    cheegers84 = pickle.load(f)

with open("data_witnesses84.pkl", "rb") as f:
    witnesses84 = pickle.load(f)

cheegers84 = np.array(cheegers84, dtype=float)

################################

X = eigenvals84
y = cheegers84

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

def evaluate_model(y_true, y_pred, name="Model"):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    target_range = np.ptp(y_true)
    results = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAE / range (%)": 100 * mae / target_range if target_range > 0 else np.nan,
        "RMSE / range (%)": 100 * rmse / target_range if target_range > 0 else np.nan,
    }
    print(f"\n{name}:")
    print("-" * 40)
    for metric, value in results.items():
        print(f"{metric:<22}{value:>18.6f}")
    return results