from typing import Dict, Set, List, Tuple
from environments.env_utils import get_environment
from deepxube.environments.environment_abstract import Environment, State
from argparse import ArgumentParser
import pickle
import time


def make_labeled(env: Environment, start_state: State, depth_max: int) -> Dict[int, Set[State]]:
    """ Assumes uniform transition costs

    :return:
    """
    seen: Set[State] = {start_state}
    fifo: List[Tuple[int, State]] = [(0, start_state)]
    ctg_to_states: Dict[int, Set[State]] = dict()
    start_time = time.time()
    depth_curr: int = 0

    while len(fifo) > 0:
        depth, state = fifo.pop(0)
        if depth > depth_curr:
            print(f"Depth {depth}, Total Time: {time.time() - start_time} seconds")
            depth_curr = depth
        if depth not in ctg_to_states.keys():
            ctg_to_states[depth] = set()
        ctg_to_states[depth].add(state)

        if depth < depth_max:
            for state_next in env.expand([state])[0][0]:
                if state_next in seen:
                    continue
                seen.add(state_next)
                fifo.append((depth + 1, state_next))

    print("DONE")
    return ctg_to_states


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help="Environment name")
    parser.add_argument('--depth', type=int, required=True, help="Maximum depth of breadth-first search")
    parser.add_argument('--save', type=str, default="", help="Location to save data")
    parser.add_argument('--viz', action='store_true', default=False, help="")
    args = parser.parse_args()

    # get environment and goal state
    env, state_goal = get_environment(args.env)

    # get labeled data
    ctg_to_states: Dict[int, Set[State]] = make_labeled(env, state_goal, args.depth)
    num_states_tot: int = 0
    for depth in range(args.depth + 1):
        num_states_tot += len(ctg_to_states[depth])
        print(f"Cost-to-go: {depth}, {len(ctg_to_states[depth])} states")
    print(f"{num_states_tot} total states")

    # test positive/negative classification
    states_all: List[State] = []
    for states_val in ctg_to_states.values():
        states_all.extend(states_val)

    if len(args.save) > 0:
        print(f"Saving to {args.save}")
        pickle.dump(ctg_to_states, open(args.save, "wb"), protocol=-1)


if __name__ == "__main__":
    main()
