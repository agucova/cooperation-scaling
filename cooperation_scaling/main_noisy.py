from typing import TypedDict
from game import play_game, OPTION
import pandas as pd
from pathlib import Path


class GameRun(TypedDict):
    family: str
    model: str
    params: int
    checkpoint: str
    training_steps: int
    moves: list[tuple[OPTION, OPTION]]
    score_p1: int
    score_p2: int
    n_rounds: int
    noise: float


class FailedGameRun(TypedDict):
    model: str
    params: int
    checkpoint: str
    training_steps: int
    n_rounds: int
    noise: float
    family: str


ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = ROOT_PATH / "data"

# Pythia setup
PARAM_SIZES = [
    # Only includes the deduped models
    ("70M", 70_426_624),
    ("160M", 162_322_944),
    ("410M", 405_334_016),
    ("1B", 1_011_781_632),
    ("1.4B", 1_414_647_808),
    ("2.8B", 2_775_208_960),
    ("6.9B", 6_857_302_016),
    ("12B", 11_846_072_320),
]
TRAINING_STEP_NUMBERS = (
    # Initial steps
    # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    # Then every 1000 steps, from step1000 to step143000 (main)
    # Fix for noisy version: only go from 11000 to 143000
    range(11000, 143000, 12000)
)
TRAINING_STEPS = [(f"step{i}", i) for i in TRAINING_STEP_NUMBERS]
NOISE_VALUES = [0, 0.25]
HF_USER = "EleutherAI"
GAME_FAMILIES = {
    "Win-win": [
        [(1, 1), (4, 4)],  # JJ, JF
        [(1, 1), (2, 2)],  # FJ, FF
    ],
    "Prisoner's Dilemma": [
        [(4, 4), (3, 1)],  # JJ, JF
        [(1, 3), (2, 2)],  # FJ, FF
    ],
    "Unfair": [
        [(4, 4), (1, 3)],  # JJ, JF
        [(1, 1), (2, 2)],  # FJ, FF
    ],
    "Biased": [
        [(4, 4), (2, 3)],  # JJ, JF
        [(1, 1), (3, 2)],  # FJ, FF
    ],
    "Second Best": [
        [(1, 1), (4, 2)],  # JJ, JF
        [(3, 3), (2, 4)],  # FJ, FF
    ],
}

if __name__ == "__main__":
    models_to_use: list[tuple[int, str]] = [
        (param_size[1], f"{HF_USER}/pythia-{param_size[0]}-deduped")
        for param_size in PARAM_SIZES
    ]
    training_steps_to_use = TRAINING_STEPS
    # Attempt to read existing game results if they exist
    games_file_path = DATA_PATH / "games_noisy.csv"
    failed_games_file_path = DATA_PATH / "failed_games_noisy.csv"

    if games_file_path.exists():
        print("Resuming from games checkpoint")
        games_df = pd.read_csv(games_file_path)
        games: list[GameRun] = games_df.to_dict("records")  # type: ignore
    else:
        games = []

    if failed_games_file_path.exists():
        print("Resuming from failed games checkpoint")
        failed_games_df = pd.read_csv(failed_games_file_path)
        failed_games: list[FailedGameRun] = failed_games_df.to_dict("records")  # type: ignore
    else:
        failed_games = []

    completed_runs = set((game["model"], game["checkpoint"]) for game in games) | set(
        (failed_game["model"], failed_game["checkpoint"])
        for failed_game in failed_games
    )

    # Run every combination of models and training steps
    for param_size, model in models_to_use:
        for checkpoint, training_steps in training_steps_to_use:
            for noise in NOISE_VALUES:
                for family_name, payoff_matrix in GAME_FAMILIES.items():
                    if (model, checkpoint) in completed_runs:
                        continue
                    print(f"Running {model} with {training_steps} training steps")
                    result = play_game(
                        (model, checkpoint),
                        (model, checkpoint),
                        "Option J",
                        "Option F",
                        payoff_matrix,
                        5,
                        noise=noise,
                    )
                    if not isinstance(result, int):
                        # Game completed successfully
                        games.append(
                            {
                                "model": model,
                                "params": param_size,
                                "checkpoint": checkpoint,
                                "training_steps": training_steps,
                                "moves": result[0],
                                "score_p1": result[1][0],
                                "score_p2": result[1][1],
                                "n_rounds": 5,
                                "noise": noise,
                                "family": family_name,
                            }
                        )
                        # Save results to disk
                        pd.DataFrame(games).to_csv(games_file_path, index=False)
                    else:
                        # Game could not be completed
                        failed_games.append(
                            {
                                "model": model,
                                "params": param_size,
                                "checkpoint": checkpoint,
                                "training_steps": training_steps,
                                "n_rounds": 5,
                                "noise": noise,
                                "family": family_name,
                            }
                        )
                        pd.DataFrame(failed_games).to_csv(
                            failed_games_file_path, index=False
                        )
