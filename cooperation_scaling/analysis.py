import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.animation as animation


ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = ROOT_PATH / "data"

games = pd.read_csv(DATA_PATH / "games.csv")

# Compute (score_p1 + score_p2) / (20 * n_rounds)
games["efficiency"] = (games["score_p1"] + games["score_p2"]) / (8 * 5)

# Plot params vs efficiency (scatterplot)
fig, ax = plt.subplots()
sns.scatterplot(data=games, x="params", y="efficiency", ax=ax)
ax.set_xlabel("Number of parameters")
ax.set_ylabel("Efficiency")
ax.set_xscale("log")
plt.savefig(ROOT_PATH / "plots" / "param_size_vs_efficiency_scatterplot.png")

# Plot training_steps vs efficiency (scatterplot)
fig, ax = plt.subplots()
sns.scatterplot(data=games, x="training_steps", y="efficiency", ax=ax)
ax.set_xlabel("Number of training steps")
ax.set_ylabel("Efficiency")
plt.savefig(ROOT_PATH / "plots" / "training_steps_vs_efficiency_scatterplot.png")

# Noisy versions
noisy_games = pd.read_csv(DATA_PATH / "games_noisy.csv")
noisy_games["efficiency"] = (noisy_games["score_p1"] + noisy_games["score_p2"]) / (8 * 5)

# Per each `params`, find the rows with the highest `training_steps`
i_max_steps = noisy_games.groupby("params")["training_steps"].idxmax()
noisy_games_max_steps = noisy_games.loc[i_max_steps]

# Plot `checkpoint` vs efficiency (scatterplot)
# with `family` as hue
fig, ax = plt.subplots()
sns.scatterplot(data=noisy_games, x="training_steps", y="efficiency", hue="family", ax=ax)
ax.set_xlabel("Params")
ax.set_ylabel("Efficiency")
plt.savefig(ROOT_PATH / "plots" / "noisy" / "params_vs_efficiency_scatterplot2.png")


# 3D scatterplot of `params`, `training_steps`, `efficiency`, hue=`family`
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Draw points for each family
for family_name, family_df in noisy_games.groupby("family"):
    ax.scatter(
        family_df["params"],
        family_df["training_steps"],
        family_df["efficiency"],
        label=family_name,
    )

# Use scientific notation for the x and y ticks
ax.ticklabel_format(style='sci', scilimits=(0,0), axis='x')
ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
# Label each axis
ax.set_xlabel("Params")
ax.set_ylabel("Training steps")
ax.set_zlabel("Efficiency (%)")
ax.legend()
plt.savefig(ROOT_PATH / "plots" / "noisy" / "params_vs_training_steps_vs_efficiency_3d_scatterplot2.png")


# Calculate defection rates based on the list of tuples in `moves`
noisy_games["moves"] = noisy_games["moves"].apply(eval) # (Trusted input)
noisy_games["defection_rate_p1"] = noisy_games["moves"].apply(lambda moves: sum(move[0] == "F" for move in moves) / len(moves))
noisy_games["defection_rate_p2"] = noisy_games["moves"].apply(lambda moves: sum(move[1] == "F" for move in moves) / len(moves))

# Filter to prisoner's dilemma games
prisoners_dilemma_games = noisy_games[noisy_games["family"] == "Prisoner's Dilemma"]

# 3d scatterplot of `params`, `training_steps`, `defection_rate_p1`, hue=`noise`
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Draw points for each noise value
for noise_value, noise_df in prisoners_dilemma_games.groupby("noise"):
    ax.scatter(
        noise_df["params"],
        noise_df["training_steps"],
        noise_df["defection_rate_p1"],
        label=f"{noise_value:.2f}",
    )

# Use scientific notation for the x and y ticks
ax.ticklabel_format(style='sci', scilimits=(0,0), axis='x')
ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
# Label each axis
ax.set_xlabel("Params")
ax.set_ylabel("Training steps")
ax.set_zlabel("Defection rate")
ax.legend()
plt.savefig(ROOT_PATH / "plots" / "noisy" / "params_vs_training_steps_vs_defection_rate_p1_3d_scatterplot2.png")