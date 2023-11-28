from typing import Generator, TypeAlias, Literal
from torch import device, cuda
from contextlib import contextmanager
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from pathlib import Path
from functools import cache
from dataclasses import dataclass
from prompts import (
    game_prompt,
    completion_to_option,
    insist_on_answer_prompt,
)
import random

ROOT_PATH = Path(__file__).parent.parent

assert cuda.is_available()

@contextmanager
def get_model_and_tokenizer(model_id: str, revision: str, cache_dir: Path = ROOT_PATH / ".model_cache") -> Generator[tuple[GPTNeoXForCausalLM, AutoTokenizer], None, None]:
    cache_dir = cache_dir / model_id / revision
    model = GPTNeoXForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        device_map="balanced_low_0"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, cache_dir=cache_dir, padding_side="left",
    )

    try:
        yield model, tokenizer # type: ignore
    finally:
        # Clear the cache
        cuda.empty_cache()


def prompt_model(prompt: str, model, tokenizer) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    tokens = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)  # type: ignore
    return tokenizer.decode(tokens[0])


@dataclass
class Player:
    id: int
    model: GPTNeoXForCausalLM
    tokenizer: AutoTokenizer

    def prompt(self, input: str) -> str:
        return prompt_model(input, self.model, self.tokenizer)


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
    noise: float = 0.0,
) -> tuple[list[tuple[OPTION, OPTION]], tuple[int, int]] | int:
    assert 0 <= noise <= 1
    assert len(payoff_matrix) == 2
    assert n_rounds > 0
    
    with get_model_and_tokenizer(model_1[0], model_1[1]) as (model, tokenizer):
        # Initialize players
        player_1 = Player(1, model, tokenizer)
        player_2 = Player(2, model, tokenizer)

        # Initialize game state
        moves: list[tuple[OPTION, OPTION]] = []
        player_1_points = 0
        player_2_points = 0
        
        for round in range(n_rounds):
            # Add noise to moves
            prompt_1 = game_prompt(moves, 1, option_j, option_f, payoff_matrix, n_rounds, noise=(noise > 0.0))
            
            move_1 = prompt_player(player_1, prompt_1, option_j, option_f)
            # Add noise to move
            if move_1 and random.random() < noise:
                move_1 = "J" if move_1 == "F" else "F"

            prompt_2 = game_prompt(moves, 2, option_j, option_f, payoff_matrix, n_rounds, noise=(noise > 0.0))
            move_2 = prompt_player(player_2, prompt_2, option_j, option_f)
            # Add noise to move
            if move_2 and random.random() < noise:
                move_2 = "J" if move_1 == "F" else "F"

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
