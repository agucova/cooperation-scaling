import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = ROOT_PATH / "data"

games = pd.read_csv(DATA_PATH / "games.csv")

# Compute (score_p1 + score_p2) / (20 * n_rounds)
games["efficiency"] = (games["score_p1"] + games["score_p2"]) / (20 * games["n_rounds"])

# Plot params vs efficiency (scatterplot)
fig, ax = plt.subplots()
sns.scatterplot(data=games, x="params", y="efficiency", ax=ax)
ax.set_xlabel("Number of parameters")
ax.set_ylabel("Efficiency")
plt.savefig(ROOT_PATH / "plots" / "param_size_vs_efficiency_scatterplot.png")

# Plot training_steps vs efficiency (scatterplot)
fig, ax = plt.subplots()
sns.scatterplot(data=games, x="training_steps", y="efficiency", ax=ax)
ax.set_xlabel("Number of training steps")
ax.set_ylabel("Efficiency")
plt.savefig(ROOT_PATH / "plots" / "training_steps_vs_efficiency_scatterplot.png")