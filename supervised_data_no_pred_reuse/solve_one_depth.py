import os
import argparse
from popper.util import Settings, order_prog, format_rule
from popper.loop import learn_solution


def write_prog_on_file(settings, depth, prog, score, filename="results.txt"):
    with open(filename, "a") as f:
        tp, fn, tn, fp, size = score
        precision = 'n/a'
        if (tp+fp) > 0:
            precision = f'{tp / (tp+fp):0.2f}'
        recall = 'n/a'
        if (tp+fn) > 0:
            recall = f'{tp / (tp+fn):0.2f}'
        f.write('*'*10 + f' DEPTH {depth} ' + '*'*10+'\n')
        if settings.noisy:
            f.write(f'Precision:{precision} Recall:{recall} TP:{tp} FN:{fn} TN:{tn} FP:{fp} Size:{size} MDL:{size+fn+fp}\n')
        else:
            f.write(f'Precision:{precision} Recall:{recall} TP:{tp} FN:{fn} TN:{tn} FP:{fp} Size:{size}\n')
        for rule in order_prog(prog):
            f.write(format_rule(settings.order_rule(rule))+'\n')
        f.write('*'*30+'\n\n')

def solve_task(dir, depth, main_results_file="results.txt", timeout=1200):
    path = os.path.join(dir, str(depth))
    settings = Settings(kbpath=path, timeout=timeout)  
    prog, score, stats = learn_solution(settings)
    # settings.print_prog_score(prog, score)
    if prog is not None:
        settings.print_prog_score(prog, score)
        write_prog_on_file(settings, depth, prog, score, main_results_file)
    else:
        with open(main_results_file, "a") as f:
            f.write('*' * 10 + f' DEPTH {depth} ' + '*' * 10 + '\n')
            f.write("NO SOLUTION\n")
            f.write('*' * 30 + '\n\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--main_results_file', type=str, default="results.txt")
    parser.add_argument('--timeout', type=int, default=1200, help='Timeout in seconds for the solver')
    args = parser.parse_args()

    solve_task(args.dir, args.depth, args.main_results_file, args.timeout)