#!/usr/bin/env python3
from sys import exit

from multilanguage import Env, Lang, TALcolors

import pirellone_lib as pl
from model_utils import ModellingProblemHelper
from utils_services import process_user_sol, print_separator


# METADATA OF THIS TAL_SERVICE:
problem="model_pirellone"
service="try_GMPL_model"
args_list = [
    ('display_output',bool),
    ('display_error',bool),
    ('check_solution',bool),
    ('sol_style',str),
    ('lang',str),
]

ENV =Env(problem, service, args_list)
TAc =TALcolors(ENV)
LANG=Lang(ENV, TAc, lambda fstring: eval(f"f'{fstring}'"))


# START CODING YOUR SERVICE:
# Initialize ModellingProblemHelper
mph = ModellingProblemHelper()
# Get input
try:
    TAc.print(LANG.render_feedback("start", f"# Hey, I am ready to start and get your input files (mod=your_mod_file.mod dat=your_dat_file.dat input=your_input_file.txt)."), "yellow")
    if ENV['check_solution']:
        input_str = mph.receive_modelling_files(read_input=True)
        instance = pl.get_pirellone_from_str(input_str)
    else:
        mph.receive_modelling_files(read_input=False)
        instance = None
except RuntimeError as err:
    err_name = err.args[0]
    # manage custom exceptions:
    if err_name == 'write-error':
        TAc.print(LANG.render_feedback('write-error', f"Fail to create {err.args[1]} file"), "red", ["bold"])
    else:
        TAc.print(LANG.render_feedback('unknown-error', f"Unknown error: {err_name}"), "red", ["bold"])
    exit(0)

# Perform solution with GPLSOL
try:
    mph.run_GPLSOL()
except RuntimeError as err:
    err_name = err.args[0]
    # manage custom exceptions:
    if err_name == 'process-timeout':
        TAc.print(LANG.render_feedback('process-timeout', "Too much computing time! Deadline exceeded."), "red", ["bold"])
    elif err_name == 'process-call':
        TAc.print(LANG.render_feedback('process-call', "The call to glpsol on your .dat file returned error."), "red", ["bold"])
    elif err_name == 'process-exception':
        TAc.print(LANG.render_feedback('process-exception', f"Processing returned with error:\n{err.args[1]}"), "red", ["bold"])
    else:
        TAc.print(LANG.render_feedback('unknown-error', f"Unknown error: {err_name}"), "red", ["bold"])
    exit(0)

# print GPLSOL output
if ENV['display_output']:
    print_separator(TAc, LANG)
    try:
        gplsol_output = mph.get_out_str()
        TAc.print(LANG.render_feedback("out-title", "The GPLSOL stdoutput is: "), "yellow", ["BOLD"])  
        TAc.print(LANG.render_feedback("stdout", f"{gplsol_output}"), "white", ["reverse"])
    except RuntimeError as err:
        err_name = err.args[0]
        # manage custom exceptions:
        if err_name == 'read-error':
            TAc.print(LANG.render_feedback('stdoutput-read-error', "Fail to read the output file of GPLSOL"), "red", ["bold"])
        else:
            TAc.print(LANG.render_feedback('unknown-error', f"Unknown error: {err_name}"), "red", ["bold"])
        exit(0)

# print GPLSOL output
if ENV['display_error']:
    print_separator(TAc, LANG)
    try:
        gplsol_error = mph.get_err_str()
        TAc.print(LANG.render_feedback("err-title", "The GPLSOL stderror is: "), "yellow", ["BOLD"])  
        TAc.print(LANG.render_feedback("stderr", f"{gplsol_error}"), "white", ["reverse"])
    except RuntimeError as err:
        err_name = err.args[0]
        # manage custom exceptions:
        if err_name == 'read-error':
            TAc.print(LANG.render_feedback('stderr-read-error', "Fail to read the output file of GPLSOL"), "red", ["bold"])
        else:
            TAc.print(LANG.render_feedback('unknown-error', f"Unknown error: {err_name}"), "red", ["bold"])
        exit(0)

# check GPLSOL solution
if ENV['check_solution']:
    print_separator(TAc, LANG)

    # Perform optimal solution with pirellone_lib
    opt_sol = pl.get_opt_sol(instance)
    m = len(opt_sol[0])
    n = len(opt_sol[1])

    # Print instance
    TAc.print(LANG.render_feedback("instance-title", f"The matrix {m}x{n} is:"), "yellow", ["bold"])
    TAc.print(LANG.render_feedback("instance", f"{pl.pirellone_to_str(instance)}"), "white", ["bold"])
    print_separator(TAc, LANG)

    # Extract GPLSOL solution
    try:
        # Get raw solution
        raw_sol = mph.get_raw_solution()
        # Parse the raw solution
        gplsol_sol = process_user_sol(ENV, TAc, LANG, raw_sol)
    except RuntimeError as err:
        err_name = err.args[0]
        # manage custom exceptions:
        if err_name == 'read-error':
            TAc.print(LANG.render_feedback('solution-read-error', "Fail to read the solution file of GPLSOL"), "red", ["bold"])
        else:
            TAc.print(LANG.render_feedback('unknown-error', f"Unknown error: {err_name}"), "red", ["bold"])
        exit(0)

    # Print GPLSOL solution
    TAc.print(LANG.render_feedback("sol-title", "The GPLSOL solution is:"), "yellow", ["BOLD"])
    if ENV['sol_style'] == 'seq':
        TAc.print(LANG.render_feedback("out_sol", f"{pl.seq_to_str(gplsol_sol)}"), "white", ["reverse"])
        gplsol_sol = pl.seq_to_subset(gplsol_sol, m, n)
    elif ENV['sol_style'] == 'subset':
        TAc.print(LANG.render_feedback("out_sol", f"{pl.subset_to_str(gplsol_sol)}"), "white", ["reverse"])
    print_separator(TAc, LANG)

    # Print optimal solution
    TAc.print(LANG.render_feedback("in-title", "The PirelloneLib solution is:"), "yellow", ["reverse"])
    if ENV['sol_style'] == 'seq':
        TAc.print(LANG.render_feedback("in-sol", f"{pl.seq_to_str(pl.subset_to_seq(opt_sol))}"), "green", ["bold"])
    elif ENV['sol_style'] == 'subset':
        TAc.print(LANG.render_feedback("in-sol", f"{pl.subset_to_str(opt_sol)}"), "green", ["bold"])
    print_separator(TAc, LANG)

    # Check the correctness of the user solution
    if pl.are_equiv(gplsol_sol, opt_sol):
        TAc.OK()
        TAc.print(LANG.render_feedback('correct', "This sequence turns off all lights."), "green", ["bold"])
    else:
        TAc.NO()
        TAc.print(LANG.render_feedback('not-correct', "This sequence doesn't turn off all lights see what happens using your solution"), "red", ["bold"])

exit(0)
