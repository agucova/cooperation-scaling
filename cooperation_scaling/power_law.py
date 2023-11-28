import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = ROOT_PATH / "data"

games_noisy = pd.read_csv(DATA_PATH / "games_noisy.csv")
games_noisy['efficiency'] = (games_noisy['score_p1'] + games_noisy['score_p2']) / 40

X = games_noisy[['params', 'training_steps']]
y = games_noisy['efficiency']

X_log = np.log(X)
y_log = np.log(y)

model = LinearRegression()
model.fit(X_log, y_log)

y_log_pred = model.predict(X_log)

y_pred = np.exp(y_log_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

coefficients = model.coef_
intercept = model.intercept_

print(f"Mean squared error: {mse:.2f}")
print(f"R2 score: {r2:.2f}")
print(f"Model coefficients: {coefficients}")

