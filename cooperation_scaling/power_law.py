import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset here
df = pd.read_csv('../data/data_noisy.csv')

df['efficiency'] = (df['score_p1'] + df['score_p2']) / 40

X = df[['params', 'training_steps']]
y = df['efficiency']

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

