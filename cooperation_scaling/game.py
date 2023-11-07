import sys
from typing import Optional, TypeAlias, Literal
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from pathlib import Path
from functools import cache
from dataclasses import dataclass
from prompts import game_prompt, completion_to_option, insist_on_answer_prompt

ROOT_PATH = Path(__file__).parent.parent


@cache
def get_model_and_tokenizer(model_id: str, revision: str, cache_dir: Path):
    model = GPTNeoXForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
    )

    return model, tokenizer


def prompt_model(prompt: str, model_id: str, revision: str) -> str:
    cache_dir = ROOT_PATH / ".model_cache" / model_id / revision
    model, tokenizer = get_model_and_tokenizer(model_id, revision, cache_dir)
    inputs = tokenizer(prompt, return_tensors="pt")
    tokens = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)  # type: ignore
    return tokenizer.decode(tokens[0])


@dataclass
class Player:
    id: int
    model: str
    revision: str

    def prompt(self, input: str) -> str:
        return prompt_model(input, self.model, self.revision)


OPTION: TypeAlias = Literal["J", "F"]


def prompt_player(player: Player, prompt: str, option_j: str, option_f: str):
    move = completion_to_option(
        player.prompt(prompt),
        option_j,
        option_f,
    )
    retry_attempts = 0
    while move is None:
        if retry_attempts > 2:
            print(f"Player {player.id} is being uncooperative. Ending game.")
            return None

        prompt += "\n" + insist_on_answer_prompt(option_j, option_f)
        move = completion_to_option(
            player.prompt(prompt),
            option_j,
            option_f,
        )
        retry_attempts += 1
    return move


def play_game(
    model_1: tuple[str, str],
    model_2: tuple[str, str],
    option_j: str,
    option_f: str,
    payoff_matrix: list[list[tuple[int, int]]],
    n_rounds: int,
) -> tuple[list[tuple[OPTION, OPTION]], tuple[int, int]] | int:
    # Initialize players
    player_1 = Player(1, model_1[0], model_1[1])
    player_2 = Player(2, model_2[0], model_2[1])

    # Initialize game state
    moves: list[tuple[OPTION, OPTION]] = []
    player_1_points = 0
    player_2_points = 0

    for round in range(n_rounds):
        prompt_1 = game_prompt(moves, 1, option_j, option_f, payoff_matrix, n_rounds)
        move_1 = prompt_player(player_1, prompt_1, option_j, option_f)
        prompt_2 = game_prompt(moves, 2, option_j, option_f, payoff_matrix, n_rounds)
        move_2 = prompt_player(player_2, prompt_2, option_j, option_f)

        # If either player is uncooperative, end the game
        if move_1 is None or move_2 is None:
            return round

        # Update scores
        # Matrix is [[(J, J), (J, F)], [(F, J), (F, F)]]
        payoffs = payoff_matrix[move_1 == "F"][move_2 == "F"]
        player_1_points += payoffs[0]
        player_2_points += payoffs[1]

        # Save moves
        moves.append((move_1, move_2))

    return moves, (player_1_points, player_2_points)
