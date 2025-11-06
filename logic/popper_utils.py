"""This takes equivalence relation into consideration, might not be consistent with cases where predicate order matter in prolog,
In this case popper with bias will give correct predicate order so comparing should not be focused on order, but focused on if they are sematically same
If predicates had no logical build then it would matter https://www.inf.ed.ac.uk/teaching/courses/lp/2015/slides/prog4.pdf
but currently its pure relational predicates, in future if changes are made for negation, non logical build then this will not work
This is currenlty highly dependent on how we are defining clauses in 8-Puzzle assuming 1 arity predicates initialize/define/bound variable for
>1 arity predicates"""

# https://arxiv.org/pdf/2502.01232
# https://arxiv.org/pdf/2008.07912
# https://arxiv.org/pdf/2005.02259

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Literal:
    predicate: str
    arguments: tuple

    def __lt__(self, other): # Allows sorting
        return (self.predicate, self.arguments) < (other.predicate, other.arguments)

def standardize_vars(literal: Literal, var_map, var_counter):
    new_args = []
    for arg in literal.arguments:
        if arg not in var_map:
            var_map[arg] = f'V{var_counter[0]}'
            var_counter[0] += 1
        new_args.append(var_map[arg])
    return Literal(literal.predicate, tuple(new_args))

def smart_split_literals(body: str):
    parts = []
    depth = 0
    current = ''
    for char in body:
        if char == ',' and depth == 0:
            parts.append(current.strip())
            current = ''
        else:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            current += char
    if current:
        parts.append(current.strip())
    return parts

def parse_literal(lit_str: str) -> Literal:
    match = re.match(r'(\w+)\((.*?)\)', lit_str) # Works with well formed literals
    if not match:
        raise ValueError(f"Could not parse literal: {lit_str}")
    pred = match.group(1)
    args = tuple(arg.strip() for arg in match.group(2).split(','))
    return Literal(pred, args)

class ClauseSet:
    __slots__ = ['normalized_clauses', '_hash']

    def __init__(self, clauses):
        # Bind to a local method reference (faster in loops)
        norm = self._normalize_clause
        # pdb.set_trace()
        self.normalized_clauses = tuple(sorted(norm(clause.strip(".")) for clause in clauses))
        self._hash = None  # Lazy-loaded hash

    def __eq__(self, other):
        if not isinstance(other, ClauseSet):
            return NotImplemented
        return self.normalized_clauses == other.normalized_clauses

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self.normalized_clauses)
        return self._hash

    def __repr__(self):
        return f"ClauseSet({self.normalized_clauses})"

    # def _normalize_clause(self, clause: str) -> str:
    #     var_map = {}
    #     var_counter = 0
    #
    #     def replace_var(match):
    #         nonlocal var_counter
    #         var = match.group()
    #         if var not in var_map:
    #             var_map[var] = f'V{var_counter}'
    #             var_counter += 1
    #         return var_map[var]
    #
    #     clause = re.sub(r'\b[A-Z_][A-Za-z0-9_]*\b', replace_var, clause)  # In case of popper result returing this is enough adding otheres for safety
    #
    #     # remove extra space among predicates
    #     clause = re.sub(r'\),\s*', '),', clause)
    #
    #     # remove extra spaces in brackets
    #     clause = re.sub(r'\s*,\s*', ',', clause)  # commas
    #     clause = re.sub(r'\s*\(\s*', '(', clause)  # (
    #     clause = re.sub(r'\s*\)\s*', ')', clause)  # )
    #
    #     # remove`:-` space before, one space after
    #     clause = re.sub(r'\s*:-\s*', ':- ', clause)
    #
    #     # remove all other extra spaces
    #     clause = re.sub(r'\s+', ' ', clause).strip()
    #
    #     pdb.set_trace()
    #
    #     return clause

    def _normalize_clause(self, clause: str) -> str:
        head_str, body_str = clause.strip().rstrip('.').split(':-') # clause f(V0):- tile5(V1),tile3(V3),indx4(V2),inplace_from(V0,V1),onrow(V0,V3,V2)'
        head = parse_literal(head_str.strip())
        body_literals = [parse_literal(l) for l in smart_split_literals(body_str)]

        # pdb.set_trace()


        var_origin_map = {}
        for lit in body_literals: #Identify roles based on 1-arity predicates
            if len(lit.arguments) == 1:
                arg = lit.arguments[0]
                if arg not in var_origin_map:
                    var_origin_map[arg] = lit.predicate # {'V1': 'tile5', 'V3': 'tile3', 'V2': 'indx4'}

        # pdb.set_trace()


        for lit in body_literals: #Default all other variables to 'ub'
            for arg in lit.arguments:
                if arg not in var_origin_map: # {'V1': 'tile5', 'V3': 'tile3', 'V2': 'indx4', 'V0': 'ub'}
                    var_origin_map[arg] = 'ub' # ub to refer unbound, not tied to any body predicate

        # pdb.set_trace()

        canonical_var_map = {} #Canonical naming based on predicate role, {'V0': 'V_ub_0', 'V1': 'V_tile5_0', 'V3': 'V_tile3_0', 'V2': 'V_indx4_0'}
        var_counter = {} # {'ub': 1, 'tile5': 1, 'tile3': 1, 'indx4': 1}

        def canonical_var(arg):
            role = var_origin_map[arg] # for arg V0, role -> ub
            if role not in var_counter:
                var_counter[role] = 0
            if arg not in canonical_var_map:
                canonical_var_map[arg] = f"V_{role}_{var_counter[role]}"
                var_counter[role] += 1
            return canonical_var_map[arg]

        # pdb.set_trace()

        #Normalize head
        head = Literal(head.predicate, tuple(canonical_var(a) for a in head.arguments)) # V0 -> V_ub_0

        # pdb.set_trace()

        #Normalize body
        normalized_body = []
        for lit in body_literals:
            new_args = tuple(canonical_var(arg) for arg in lit.arguments) # V1 -> V_tile5_0
            normalized_body.append(Literal(lit.predicate, new_args))

        # pdb.set_trace()

        #Sort body literals for canonical order
        normalized_body.sort()

        # pdb.set_trace()

        #Reconstruct normalized clause string
        head_str = f'{head.predicate}({",".join(head.arguments)})'
        body_str = ','.join(f'{lit.predicate}({",".join(lit.arguments)})' for lit in normalized_body)

        return f'{head_str}:- {body_str}'


# c1 = 'f(V0):- tile2(V1),indx3(V2),tile4(V3),inplace_from(V0,V3),onrow(V0,V1,V2)' # Works for all four cases
# c2 = 'f(V0):- indx3(V1),tile2(V2),onrow(V0,V2,V1),tile4(V3),inplace_from(V0,V3)'
#
# c1 = 'f(V0):- tile2(V1),indx3(V2),tile4(V3),inplace_from(V0,V3),onrow(V0,V1,V2).'
# c2 = 'f(V0):- indx3(V2),tile2(V1),onrow(V0,V1,V2),tile4(V3),inplace_from(V0,V3).'

# c1 = 'f(V0):- tile2(V1),indx3(V2),onrow(V0,V1,V2).'
# c2 = 'f(V0):- indx3(V2),tile2(V1),onrow(V0,V1,V2).'

# c1 = 'f(V0):- onrow(V0,V1,V2),tile2(V1),indx3(V2).' # Even if order is changed for above clasue works nicely
# c2 = 'f(V0):- onrow(V0,V1,V2),indx3(V2),tile2(V1).'

# c1 = ['f(V0):- tile2(V1),indx3(V2),onrow(V0,V1,V2).']
# c2 = ['f(V0):- tile2(V2),indx3(V1),onrow(V0,V2,V1).']

# c1 = ['f(V0):- tile2(V1),indx3(V2),tile4(V3),inplace_from(V0,V3),onrow(V0,V1,V2)', 'f(V0):- tile2(V1),indx3(V2),onrow(V0,V1,V2).']
# c2 = ['f(V0):- indx3(V1),tile2(V2),onrow(V0,V2,V1),tile4(V3),inplace_from(V0,V3)', 'f(V0):- tile2(V2),indx3(V1),onrow(V0,V2,V1).']

# c1 = ['f(V0):- tile2(V1),indx3(V2),tile4(V3),inplace_from(V0,V3),onrow(V0,V1,V2)', 'f(V0):- tile2(V2),indx3(V1),onrow(V0,V2,V1).']
# c2 = ['f(V0):- indx3(V1),tile2(V2),onrow(V0,V2,V1),tile4(V3),inplace_from(V0,V3)', 'f(V0):- tile2(V1),indx3(V2),onrow(V0,V1,V2).']

# c1 = ['f(V0):- tile2(V2),indx3(V1),onrow(V0,V2,V1).', 'f(V0):- tile2(V1),indx3(V2),tile4(V3),inplace_from(V0,V3),onrow(V0,V1,V2)']
# c2 = ['f(V0):- indx3(V1),tile2(V2),onrow(V0,V2,V1),tile4(V3),inplace_from(V0,V3)', 'f(V0):- tile2(V1),indx3(V2),onrow(V0,V1,V2).']

# c1 = ['f(V0):- tile2(V1),indx3(V2),tile4(V3),inplace_from(V0,V3),onrow(V0,V1,V2)', 'f(V0):- tile2(V2),indx3(V1),onrow(V0,V2,V1).']
# c2 = ['f(V0):- tile2(V1),indx3(V2),onrow(V0,V1,V2).', 'f(V0):- indx3(V1),tile2(V2),onrow(V0,V2,V1),tile4(V3),inplace_from(V0,V3)', ]

# c1 = ['f(V0):- tile2(V1),indx3(V2),tile4(V3),inplace_from(V0,V3),onrow(V0,V1,V2)', 'f(V0):- tile2(V2),indx3(V1),onrow(V0,V2,V1).'] # False
# c2 = ['f(V0):- tile2(V1),indx3(V2),onrow(V0,V1,V2).', 'f(V0):- indx3(V1),tile2(V2),onrow(V0,V1,V2),tile4(V3),inplace_from(V0,V3)', ]

# c1 = ['f(V0):- tile4(V3),inplace_from(V0,V3),tile2(V2),indx3(V1),onrow(V0,V2,V1)']
# c2 = ['f(V0):- tile4(V2),inplace_from(V0,V2),tile2(V3),indx3(V1),onrow(V0,V3,V1)']

# c1 = ['f(V0):- tile5(V1),tile3(V3),indx4(V2),inplace_from(V0,V1),onrow(V0,V3,V2)',
#       'f(V0):- tile7(V3),tile5(V2),inplace_from(V0,V3),tile1(V5),indx2(V4),onrow(V0,V5,V4),indx6(V1),onrow(V0,V2,V1)']
#
# c2 = ['f(V0):- tile5(V2),indx4(V1),inplace_from(V0,V2),tile3(V3),onrow(V0,V3,V1)',
#       'f(V0):- tile7(V4),inplace_from(V0,V4),indx2(V1),indx6(V3),tile5(V5),onrow(V0,V5,V3),tile1(V2),onrow(V0,V2,V1)']

# c1 = ['f(V0):- tile5(V1),tile3(V3),indx4(V2),inplace_from(V0,V1),onrow(V0,V3,V2)',
#       'f(V0):- tile7(V3),tile5(V2),inplace_from(V0,V3),tile1(V5),indx2(V4),onrow(V0,V5,V4),indx6(V1),onrow(V0,V2,V1)']
#
# c2 = ['f(V0):- tile5(V2),indx4(V1),inplace_from(V0,V2),tile3(V3),onrow(V0,V3,V1)',
#       'f(V0):- tile7(V4),inplace_from(V0,V4),indx2(V1),indx6(V3),tile5(V5),onrow(V0,V5,V3),tile1(V2),onrow(V0,V2,V1)']

# c1 = ['f(V0):- tile(X), tile(Y), onrow(V0, X, Y)'] # Works for this as well not necessary here

# cs1 = ClauseSet((c1,))
# cs1 = ClauseSet((c1))
# pdb.set_trace()
# cs2 = ClauseSet((c2,))
# cs2 = ClauseSet((c2))
# pdb.set_trace()
# print(cs1 == cs2)