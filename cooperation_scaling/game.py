from typing import TypeAlias, Literal
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from pathlib import Path
from functools import cache
from dataclasses import dataclass
from prompts import game_prompt, answer_to_option

model = GPTNeoXForCausalLM.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
    revision="step3000",
    cache_dir="./pythia-70m-deduped/step3000",
)

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
    revision="step3000",
    cache_dir="./pythia-70m-deduped/step3000",
)

ROOT_PATH = Path(__file__).parent.parent


@cache
def get_model_and_tokenizer(model: str, revision: str, cache_dir: Path):
    model = GPTNeoXForCausalLM.from_pretrained(
        model,
        revision=revision,
        cache_dir=cache_dir,
    )  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        revision=revision,
        cache_dir=cache_dir,
    )

    return model, tokenizer


def prompt_model(prompt: str, model: str, revision: str) -> str:
    cache_dir = ROOT_PATH / ".model_cache" / model / revision
    model, tokenizer = get_model_and_tokenizer(model, revision, cache_dir)
    inputs = tokenizer(prompt, return_tensors="pt")
    tokens = model.generate(**inputs)  # type: ignore
    return tokenizer.decode(tokens[0])


@dataclass
class Player:
    model: str
    revision: str

    def prompt(self, input: str) -> str:
        return prompt_model(input, self.model, self.revision)


OPTION: TypeAlias = Literal["J", "F"]


def play_game(
    option_j: str,
    option_f: str,
    payoff: list[list[tuple[int, int]]],
    n_rounds: int,
):
    # Initialize players
    player_1 = Player("EleutherAI/pythia-70m-deduped", "step3000")
    player_2 = Player("EleutherAI/pythia-70m-deduped", "step3000")

    # Initialize game state
    moves: list[tuple[OPTION, OPTION]] = []
    player_1_points = 0
    player_2_points = 0

    for round in range(n_rounds):
        print("Playing round", round + 1)
        move1 = answer_to_option(
            player_1.prompt(game_prompt(moves, 1, option_j, option_f, payoff)),
            option_j,
            option_f,
        )
        move2 = answer_to_option(
            player_2.prompt(game_prompt(moves, 2, option_j, option_f, payoff)),
            option_j,
            option_f,
        )

        # Update scores
        player_1_points += payoff[move1 == "F"][move2 == "F"][0]
        player_2_points += payoff[move1 == "J"][move2 == "J"][1]

        # Save moves
        moves.append((move1, move2))
        
        print("Player 1 points:", player_1_points)
        print("Player 2 points:", player_2_points)
