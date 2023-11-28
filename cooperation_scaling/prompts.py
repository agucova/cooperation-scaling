from inspect import cleandoc
from typing import Literal, Optional
import re


def initial_prompt(
    option_j: str,
    option_f: str,
    payoff: list[list[tuple[int, int]]],
    n_rounds: int,
    player: Literal[1, 2],
) -> str:
    player_id = player - 1
    other_player_id = 1 - player_id

    return cleandoc(
        f"""
    You are playing a game repeatedly with another player. In this game, you can choose between {option_j} and {option_f}. You will play {n_rounds} rounds in total with the same player.
    The rules of the game are as follows:
    - If you choose {option_j} and the other player chooses {option_j}, then you win {payoff[0][0][player_id]} points and the other player wins {payoff[0][0][other_player_id]} points.
    - If you choose {option_j} and the other player chooses {option_f}, then you win {payoff[0][1][player_id]} points and the other player wins {payoff[0][1][other_player_id]} points.
    - If you choose {option_f} and the other player chooses {option_j}, then you win {payoff[1][0][player_id]} points and the other player wins {payoff[1][0][other_player_id]} points.
    - If you choose {option_f} and the other player chooses {option_f}, then you win {payoff[1][1][player_id]} points and the other player wins {payoff[1][1][other_player_id]} points.
    """
    )


def move_to_text(move: Literal["J", "F"], option_j: str, option_f: str) -> str:
    if move == "J":
        return option_j
    elif move == "F":
        return option_f
    else:
        raise ValueError("move must be either J or F")


def round_prompt(
    moves: list[tuple[Literal["J", "F"], Literal["J", "F"]]],
    option_j: str,
    option_f: str,
    payoff: list[list[tuple[int, int]]],
    player: Literal[1, 2],
) -> str:
    player_id = player - 1
    other_player_id = 1 - player_id

    return cleandoc(
        f"""
        In round {len(moves)}, you chose {move_to_text(moves[-1][player_id], option_j, option_f)} and the other player chose {move_to_text(moves[-1][other_player_id], option_j, option_f)}.
        Thus, you won {payoff[moves[-1][player_id] == "J"][moves[-1][other_player_id] == "J"][player_id]} points and the other player won {payoff[moves[-1][player_id] == "J"][moves[-1][other_player_id] == "J"][other_player_id]} points.
        """
    )


def select_option_prompt(
    current_round: int,
    option_j: str,
    option_f: str,
):
    return cleandoc(
        f"""
            You are currently playing round {current_round}.
            Q: Which option do you choose, '{option_j}' or '{option_f}'? (type either option, without any additional text)
            A: """
    )


NOISE_PROMPT = "Be aware that the other player can make mistakes. "


def game_prompt(
    moves: list[tuple[Literal["J", "F"], Literal["J", "F"]]],
    player: Literal[1, 2],
    option_j: str,
    option_f: str,
    payoff: list[list[tuple[int, int]]],
    n_rounds: int,
    noise: bool = False,
):
    """
    Generate full game prompt for one player in this round.
    """

    if len(moves) == 0:
        return (
            initial_prompt(option_j, option_f, payoff, n_rounds, player)
            + (f"\n{NOISE_PROMPT}"
            if noise
            else "") + "\n" + select_option_prompt(1, option_j, option_f)
        )
    else:
        header = initial_prompt(option_j, option_f, payoff, n_rounds, player)
        previous_rounds = "\n".join(
            [
                round_prompt(
                    moves[: i + 1],
                    option_j,
                    option_f,
                    payoff,
                    player,
                )
                for i in range(len(moves))
            ]
        )
        current_round = select_option_prompt(
            len(moves) + 1,
            option_j,
            option_f,
        )
        return cleandoc(
            header + "\n" + (f"{NOISE_PROMPT}\n"
            if noise
            else "") + previous_rounds + "\n" + current_round
        )



def completion_to_option(
    answer: str, option_j: str, option_f: str
) -> Optional[Literal["J", "F"]]:
    print(f"Answer: {answer}")
    # Get text after the last "A:" in the answer
    response = answer.split("A:")[-1]
    print(f"Response: {response}")
    # Strip, lowercase, and remove punctuation
    response = response.strip().lower().replace(".", "").replace(",", "")

    if re.match(r"\boption\s+j\b", response) or re.match(r"\bj\b", response):
        print(f"Matched {response} to {option_j}")
        return "J"
    elif re.match(r"\boption\s+f\b", response) or re.match(r"\bf\b", response):
        print(f"Matched {response} to {option_f}")
        return "F"
    else:
        print(f"Could not match: {response}")
        return None

def insist_on_answer_prompt(
    option_j: str,
    option_f: str,
):
    return cleandoc( 
        f"""
            Invalid answer. Please answer exactly either '{option_j}' or '{option_f}'.
            Q: Which option do you choose, '{option_j}' or '{option_f}'?
            A: """
    )


if __name__ == "__main__":
    # Test the prompts
    # Initial prompt
    print(
        game_prompt(
            [],
            1,
            "J",
            "F",
            [
                [(3, 3), (0, 5)],
                [(5, 0), (1, 1)],
            ],
            5,
        )
    )
    # Round 1
    print(
        game_prompt(
            [("J", "J")],
            1,
            "J",
            "F",
            [
                [(3, 3), (0, 5)],
                [(5, 0), (1, 1)],
            ],
            5,
        )
    )
    # Round 2
    print(
        game_prompt(
            [("J", "J"), ("J", "F")],
            1,
            "J",
            "F",
            [
                [(3, 3), (0, 5)],
                [(5, 0), (1, 1)],
            ],
            5,
        )
    )