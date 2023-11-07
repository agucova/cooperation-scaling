from typing import TypedDict
from game import play_game, OPTION
import pandas as pd
from pathlib import Path


class GameRun(TypedDict):
    model: str
    params: int
    checkpoint: str
    training_steps: int
    moves: list[tuple[OPTION, OPTION]]
    score_p1: int
    score_p2: int
    n_rounds: int


class FailedGameRun(TypedDict):
    model: str
    params: int
    checkpoint: str
    training_steps: int
    n_rounds: int


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
    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    # Then every 1000 steps, from step1000 to step143000 (main)
    + [1000 + 1000 * i for i in range(143)]
)
TRAINING_STEPS = [(f"step{i}", i) for i in TRAINING_STEP_NUMBERS]
HF_USER = "EleutherAI"


if __name__ == "__main__":
    models_to_use: list[tuple[int, str]] = [
        (param_size[1], f"{HF_USER}/pythia-{param_size[0]}-deduped")
        for param_size in PARAM_SIZES
    ]
    # Choose a subset of the TRAINING_STEPS to use
    training_steps_to_use = [
        TRAINING_STEPS[i] for i in range(0, len(TRAINING_STEPS), 10)
    ]

    # Attempt to read existing game results if they exist
    games_file_path = DATA_PATH / "games.csv"
    failed_games_file_path = DATA_PATH / "failed_games.csv"

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
            if (model, checkpoint) in completed_runs:
                continue
            print(f"Running {model} with {training_steps} training steps")
            result = play_game(
                (model, checkpoint),
                (model, checkpoint),
                "Option J",
                "Option F",
                [
                    [(3, 3), (0, 5)],
                    [(5, 0), (1, 1)],
                ],
                5,
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
                        "n_rounds": result,
                    }
                )
                pd.DataFrame(failed_games).to_csv(failed_games_file_path, index=False)