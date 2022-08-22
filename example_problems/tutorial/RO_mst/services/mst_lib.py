import ast, math, operator, itertools
from sys import stderr
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass
from numpy import true_divide
import networkx as nx
from RO_verify_submission_gen_prob_lib import verify_submission_gen

instance_object_specs = {
    ('n', int),
    ('s', int),
    ('t', int),
    ('edges', str),
    ('forbidden_edges', str),
    ('forced_edges', str),
    ('query_edge', str),
}

answer_objects_spec = {
    'opt_sol': str,
    'opt_val': str,
    'num_opt_sols': int,
    'list_opt_sols': str,
    'edge_profile': str,
    # 'cyc_cert': str,
    # 'edgecut_cert': str,
    # 'cutshore_cert': str,
}

answer_objects_implemented = [
    'opt_sol',
    'opt_val',
    'num_opt_sols',
    'list_opt_sols',
    'edge_profile',
    # 'cyc_cert',
    # 'edgecut_cert',
    # 'cutshore_cert',
]


def solver(input_to_oracle):
    # inserire il solver qui
    INSTANCE = input_to_oracle["instance"]
    # TODO


class verify_submission_problem_specific(verify_submission_gen):
    def __init__(self, SEF, input_data_assigned: Dict, long_answer_dict: Dict, oracle_response: Dict = None):
        super().__init__(SEF, input_data_assigned, long_answer_dict, oracle_response)

    def verify_format(self, SEF):
        if not super().verify_format(SEF):
            return False
        # TODO

    def verify_feasibility(self, SEF):
        if not super().verify_feasibility(SEF):
            return False
        # TODO

    def verify_consistency(self, SEF):
        if not super().verify_consistency(SEF):
            return False
        # TODO

    def verify_optimality(self, SEF):
        if not super().verify_optimality(SEF):
            return False
        # TODO
