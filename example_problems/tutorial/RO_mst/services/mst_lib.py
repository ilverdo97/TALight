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
    'edge_profile': str
}

answer_objects_implemented = ['opt_sol', 'opt_val', 'num_opt_sols', 'list_opt_sols', 'edge_profile']

