from typing import Tuple, cast
from environments.n_puzzle import NPuzzle, NPState
from deepxube.environments.environment_abstract import EnvGrndAtoms, State
from deepxube.environments.cube3 import Cube3, Cube3State
import re
import math


def get_environment(env_name: str) -> Tuple[EnvGrndAtoms, State]:
    env: EnvGrndAtoms
    state_goal: State
    puzzle_n_regex = re.search(r"puzzle(\d+)", env_name)
    if env_name == "cube3":
        env: EnvGrndAtoms = Cube3(env_name)
        state_goal: Cube3State = Cube3State(cast(Cube3, env).goal_colors.copy())
        # clause_str: str = ("goal :- cubelet(Cbl), color(Col), face(F), onface(Cbl, Col, F), face_col(F, FCol), "
        #                   "dif_col(Col, FCol)")
    elif puzzle_n_regex is not None:
        puzzle_dim: int = int(math.sqrt(int(puzzle_n_regex.group(1)) + 1))
        env: EnvGrndAtoms = NPuzzle(env_name, puzzle_dim)
        state_goal: NPState = NPState(cast(NPuzzle, env).goal_tiles.copy())
        # clause_str: str = "goal :- at_row(t1, r0), at_col(t1, c0)"
    else:
        raise ValueError(f"Unknown environment {env_name}")

    return env, state_goal
