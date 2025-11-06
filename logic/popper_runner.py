import os
import sys
import pickle
from typing import Dict, List
from popper.util import Settings, order_prog, format_rule
from popper.loop import learn_solution

def print_prog_score(settings: Settings, prog, score: List[int]):
    tp, fn, tn, fp, size = score
    precision = 'n/a'
    if (tp + fp) > 0:
        precision = f'{tp / (tp + fp):0.2f}'
    recall = 'n/a'
    if (tp + fn) > 0:
        recall = f'{tp / (tp + fn):0.2f}'
    # print('*' * 10 + ' SOLUTION ' + '*' * 10)
    # if settings.noisy: # TODO If noisy return this value not implemented currently
    #     print(f'Precision:{precision} Recall:{recall} TP:{tp} FN:{fn} TN:{tn} FP:{fp} Size:{size} MDL:{size + fn + fp}')
    # else:
    #     print(f'Precision:{precision} Recall:{recall} TP:{tp} FN:{fn} TN:{tn} FP:{fp} Size:{size}')
    # print(self.format_prog(order_prog(prog)))
    clauses = []
    for rule in order_prog(prog):
        # print(format_rule(settings.order_rule(rule)))
        clauses.append(format_rule(settings.order_rule(rule)).strip('.'))

    # print('*' * 30)

    return clauses, precision, recall, tp, fn, tn, fp, size

# def run_popper(positive_states: List[State], negative_states: List[State], depth: int, depth_folder: str):
def run_popper(depth: int, depth_folder: str):
    # Run Popper
    settings = Settings(kbpath=depth_folder, max_vars=6, max_body=8)
    prog, score, _ = learn_solution(settings)

    clauses_dict: Dict = {depth: []}

    with open(os.path.join(depth_folder, "solution.txt"), "w") as f:
        f.write(f"********** DEPTH {depth} **********\n")
        if prog is not None:
            # settings.print_prog_score(prog, score)
            clauses, precision, recall, tp, fn, tn, fp, size = print_prog_score(settings, prog, score)
            f.write(f"Precision:{precision} Recall:{recall} TP:{tp} FN:{fn} TN:{tn} FP:{fp} Size:{size}\n")
            for clause in clauses:
                f.write(f"{clause}\n")
                clauses_dict[depth].append(clause)
            # pdb.set_trace()
        else:
            f.write("NO SOLUTION\n")
        f.write("******************************\n")

    # Save to clauses.pkl
    with open(os.path.join(depth_folder, "clauses_temp.pkl"), "wb") as f:
        pickle.dump(clauses_dict, f)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python popper_runner.py <depth:int> <depth_folder:path>")
        sys.exit(1)

    depth_arg = int(sys.argv[1])
    folder_arg = sys.argv[2]

    run_popper(depth_arg, folder_arg)
