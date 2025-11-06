# File: test_prolog_heur.py
import os
import re
from pyswip import Prolog # Ensure PrologError might be needed if specific error handling is desired
from typing import List, Dict, Optional, Callable, Any # Add Any if used

import numpy as np # Added for NDArray if used by HeurComb
from numpy.typing import NDArray # Added for NDArray if used by HeurComb


from deepxube.environments.environment_abstract import EnvGrndAtoms, State, Goal # Ensure types are available

# Module-level constant for default BK filename
DEFAULT_BK_FILE_NAME = "bk.pl"

class PrologRuleChecker:
    def check_rules(self, combined_bk_filepath: str, learned_clauses: List[str], prolog_state: str) -> bool:
        """
        Checks if a state satisfies ANY of the learned_clauses for a specific depth context.
        The learned_clauses (e.g., [ "f(S):-body1.", "f(S):-body2." ]) are asserted with a
        temporary head. Prolog's OR semantics naturally apply when querying this temporary head.
        """
        prolog = Prolog() # Fresh instance
        try:
            abs_bk_filepath = os.path.abspath(combined_bk_filepath)
            if not os.path.exists(abs_bk_filepath):
                print(f"!!! ERROR in PrologRuleChecker: BK file not found: {abs_bk_filepath}")
                return False
            
            # Optional: Suppress verbose loading messages if desired
            # list(prolog.query("set_prolog_flag(verbose_load, false)."))
            
            consult_query = f"consult('{abs_bk_filepath}')"
            # print(f"    PrologRuleChecker: Consulting BK: {abs_bk_filepath}") # Debug
            list(prolog.query(consult_query))

            temp_check_pred = "check_this_state_dyn" 
            # Ensure the temporary predicate is clean for this check
            # abolish/1 removes the predicate and its clauses.
            list(prolog.query(f"abolish({temp_check_pred}/1)")) 

            if not learned_clauses: # No rules for this depth, so it cannot be true
                return False

            for clause_str_from_popper in learned_clauses:
                # Ensure clause_str_from_popper is a string
                if not isinstance(clause_str_from_popper, str):
                    print(f"Warning: Non-string clause encountered: {clause_str_from_popper}")
                    continue

                if ":-" in clause_str_from_popper:
                    head_orig, body_orig = clause_str_from_popper.split(":-", 1)
                    body_cleaned = body_orig.strip().removesuffix('.')
                else: 
                    head_orig = clause_str_from_popper.strip().removesuffix('.')
                    body_cleaned = "" 
                
                # Replace the original head (f, goal, clause) with the temporary predicate
                temp_head = re.sub(r"^\s*(f|goal|clause)\s*\(\s*([A-Za-z_0-9]+)\s*\)", rf"{temp_check_pred}(\2)", head_orig.strip())
                
                if body_cleaned:
                    clause_term_for_assert = f"({temp_head} :- {body_cleaned})" 
                else: # It's a fact
                    clause_term_for_assert = temp_head
                
                # print(f"      PrologRuleChecker: Asserting term: {clause_term_for_assert}") # Debug
                prolog.assertz(clause_term_for_assert)

            query_to_execute = f"{temp_check_pred}({prolog_state})"
            # print(f"    PrologRuleChecker: Executing query: {query_to_execute}") # Debug
            result = bool(list(prolog.query(query_to_execute)))
            
            # Clean up the temporary predicate from this specific check
            list(prolog.query(f"abolish({temp_check_pred}/1)"))
            # Optional: Restore verbose loading if it was changed
            # list(prolog.query("set_prolog_flag(verbose_load, true)."))
            return result
        # except pyswip.PrologError as e: # More specific error handling
        #     print(f"!!! PrologError during check_rules for state '{prolog_state[:50]}...' using BK '{combined_bk_filepath}'.")
        #     print(f"    Prolog System Error: {e}")
        #     # print(f"    Problematic clauses: {learned_clauses}") # Debug
        #     return False
        except Exception as e: 
            print(f"!!! Unexpected Python error during Prolog check for state '{prolog_state[:50]}...' using BK '{combined_bk_filepath}': {type(e).__name__} - {e}")
            # print(f"    Problematic clauses: {learned_clauses}") # Debug
            return False

class HeurILP:
    DEFAULT_BK_FILE_NAME = "bk.pl"

    def __init__(self, env: EnvGrndAtoms, popper_run_base_dir: str):
        self.env: EnvGrndAtoms = env
        self.popper_run_base_dir = popper_run_base_dir
        # Stores rules: Dict[float_depth, List_of_rule_strings]
        self.heur: Dict[float, List[str]] = dict()
        # Sorted list of depths for which rules exist
        self.ctgs_sorted: List[float] = []
        self.checker = PrologRuleChecker()

    def add(self, ctg: float, rule: str):
        """Adds a rule string for a given ctg/depth."""
        if ctg not in self.heur:
            self.heur[ctg] = []
        if rule not in self.heur[ctg]: # Ensure rule uniqueness for this ctg
            self.heur[ctg].append(rule)
        # ctgs_sorted will be updated in get_heur or when load_rules is called externally

    def clear(self):
        """Clears all learned rules and sorted depths."""
        self.heur.clear()
        self.ctgs_sorted.clear()

    def get_heur(self, states: List[State]) -> List[Optional[float]]:
        """
        Calculates heuristic values for a list of states using learned ILP rules.
        Implements binary search to find the max depth 'd' for which rules_d(state) holds.
        """
        # Ensure ctgs_sorted is up-to-date if rules were added incrementally
        if len(self.ctgs_sorted) != len(self.heur.keys()):
             self.ctgs_sorted = sorted([k for k in self.heur.keys() if self.heur[k]]) # Only include depths with rules

        heurs: List[Optional[float]] = []
        if not self.ctgs_sorted: # No rules learned yet
            return [None] * len(states)

        states_prolog_strings: List[str] = self.env.state_to_prolog_list(states)
        
        for state_prolog_str in states_prolog_strings:
            heurs.append(self._get_heur_for_single_state_binary_search(state_prolog_str))
        return heurs

    def _get_heur_for_single_state_binary_search(self, state_prolog_str: str) -> Optional[float]:
        """
        Performs a binary search on self.ctgs_sorted to find the maximum depth (ctg_key)
        for which the state satisfies the learned rules.
        Assumes rules_d(S) => rules_{d-1}(S) (monotonicity).
        """
        low = 0
        high = len(self.ctgs_sorted) - 1
        max_satisfied_depth: Optional[float] = None

        while low <= high:
            mid_idx = (low + high) // 2
            current_depth_to_check = self.ctgs_sorted[mid_idx]
            rules_for_this_depth = self.heur.get(current_depth_to_check, [])

            # Determine the BK file: Rules for 'current_depth_to_check' were learned
            # by Popper using a BK file that contained rules from depths *less than*
            # 'current_depth_to_check'. This BK file is located in the
            # popper_run_dir for 'current_depth_to_check'.
            bk_file_to_consult = os.path.join(self.popper_run_base_dir, 
                                              str(int(current_depth_to_check)), 
                                              self.DEFAULT_BK_FILE_NAME)
            
            # print(f"  Binary Search: Checking depth {current_depth_to_check} for state {state_prolog_str[:30]}...") # Debug

            if not os.path.exists(bk_file_to_consult):
                print(f"Warning (HeurILP._get_heur_for_single_state_binary_search): "
                      f"BK file {bk_file_to_consult} not found for checking depth {current_depth_to_check}. "
                      f"This depth will be treated as not satisfied.")
                # This case means we can't prove rules for this depth.
                # To maintain the search for max_d where f_d(S) holds,
                # if we can't check f_mid(S), we assume f_mid(S) is false.
                # Thus, we need to search in lower depths.
                high = mid_idx - 1
                continue

            if rules_for_this_depth: # Only check if there are rules for this depth
                if self.checker.check_rules(bk_file_to_consult, rules_for_this_depth, state_prolog_str):
                    # If rules for this depth are satisfied, this depth is a potential answer.
                    # We try to find a GREATER depth that also satisfies its rules.
                    max_satisfied_depth = current_depth_to_check
                    low = mid_idx + 1 
                else:
                    # Rules for this depth are not satisfied.
                    # We need to check a SMALLER depth.
                    high = mid_idx - 1
            else:
                # No rules for this depth, so it's not satisfied. Search lower.
                high = mid_idx - 1
                
        # print(f"  Binary Search Result for state {state_prolog_str[:30]}...: {max_satisfied_depth}") # Debug
        return max_satisfied_depth


class HeurComb: # Assuming HeurComb might still be used elsewhere or for other purposes
    def __init__(self, heur_ilp: HeurILP, heur_fn: Callable[[List[State], List[Goal]], NDArray[np.float64]]):
        self.heur_ilp: HeurILP = heur_ilp
        self.heur_fn: Callable[[List[State], List[Goal]], NDArray[np.float64]] = heur_fn

    def get_heur(self, states: List[State], goals: List[Goal]) -> NDArray[np.float32]: # Changed to NDArray output
        # Get ILP heuristics
        # Note: self.heur_ilp.get_heur now returns List[Optional[float]]
        ilp_heur_values_optional: List[Optional[float]] = self.heur_ilp.get_heur(states)

        # Convert to numpy array, handling Nones. Let's assume Nones become 0 or a very low number
        # for the combination, or are handled by the neural net part.
        # If None means "ILP couldn't determine", neural net takes over.
        # If None means "ILP determined 0", that's different.
        # The binary search now returns None if no depth's rules are satisfied.
        
        processed_ilp_heurs = np.array([val if val is not None else -1.0 for val in ilp_heur_values_optional], dtype=np.float32)
        # Using -1.0 as a placeholder for "not found by ILP" / "needs nnet"
        # print(f"HeurComb: ILP heurs ( اولیه ): {processed_ilp_heurs}") # Debug (Persian for initial)

        none_idxs = np.where(processed_ilp_heurs == -1.0)[0]
        
        final_heurs = processed_ilp_heurs.copy()

        if len(none_idxs) > 0:
            states_none: List[State] = [states[idx] for idx in none_idxs]
            goals_none: List[Goal] = [goals[idx] for idx in none_idxs]
            
            if states_none: # Ensure there are states to pass to the neural net
                # print(f"HeurComb: Calling nnet for {len(states_none)} states.") # Debug
                nnet_heurs: NDArray[np.float64] = self.heur_fn(states_none, goals_none) # Expects specific input
                final_heurs[none_idxs] = nnet_heurs.astype(np.float32)
            # else: # Debug
                # print("HeurComb: No states needed nnet (all had ILP heurs or none_idxs was empty initially).")


        # print(f"HeurComb: Final heurs: {final_heurs}") # Debug
        # Ensure output is a simple 1D array of floats as expected by AStar after processing
        return final_heurs.astype(np.float32)