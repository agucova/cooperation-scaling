from typing import Literal
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

    return f"""You are playing a game repeatedly with another
    player. In this game, you can choose between {option_j} and {option_f}. 
    You will play {n_rounds} rounds in total with the same player.
    The rules of the game are as follows:
    - If you choose {option_j} and the other player chooses
    {option_j}, then you win {payoff[0][0][player_id]} points and the other
    player wins {payoff[0][0][other_player_id]} points.
    - If you choose {option_j} and the other player chooses
    {option_f}, then you win {payoff[0][1][player_id]} points and the other player
    wins {payoff[0][1][other_player_id]} points.
    - If you choose {option_f} and the other player chooses
    {option_j}, then you win {payoff[1][0][player_id]} points and the other player
    wins {payoff[1][0][other_player_id]} points.
    - If you choose {option_f} and the other player chooses
    {option_f}, then you win {payoff[1][1][player_id]} points and the other player
    wins {payoff[1][1][other_player_id]} points.
    """


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

    return f"""In round {len(moves)}, you chose {move_to_text(moves[-1][player_id], option_j, option_f)}
        and the other player chose {move_to_text(moves[-1][other_player_id], option_j, option_f)}.
        Thus, you won {payoff[moves[-1][player_id] == "J"][moves[-1][other_player_id] == "J"][player_id]} points and the other player won {payoff[moves[-1][player_id] == "J"][moves[-1][other_player_id] == "J"][other_player_id]} points.
        """


def select_option_prompt(
    current_round: int,
    option_j: str,
    option_f: str,
):
    return f"""You are currently plaing round {current_round}.
            Q: Which Option do you choose, {option_j} or {option_f}?
            A: """

def game_prompt(
    moves: list[tuple[Literal["J", "F"], Literal["J", "F"]]],
    player: Literal[1, 2],
    option_j: str,
    option_f: str,
    payoff: list[list[tuple[int, int]]],
):
    """
    Generate full game prompt for one player in this round.
    """
    if len(moves) == 0:
        return initial_prompt(option_j, option_f, payoff, len(moves), player)
    else:
        header = initial_prompt(option_j, option_f, payoff, len(moves), player)
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
        return header + previous_rounds + current_round
    
def answer_to_option(answer: str, option_j: str, option_f: str) -> Literal["J", "F"]:
    # Strip, lowercase, and remove punctuation
    answer = answer.strip().lower().replace(".", "").replace(",", "")
    # Use regex to match the answer to the option
    if re.match(rf"\b{option_j}\b", answer):
        return "J"
    elif re.match(rf"\b{option_f}\b", answer):
        return "F"
    else:
        raise ValueError("answer must be either J or F")