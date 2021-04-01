#!/usr/bin/env python3
from sys import stderr, exit, argv

from multilanguage import Env, Lang, TALcolors
from TALinputs import TALinput

from parentheses_lib import recognize, Par

# METADATA OF THIS TAL_SERVICE:
problem="parentheses"
service="check_rank"
args_list = [
    ('input_formula',str),
    ('rank',int),
    ('sorting_criterion',str),
    ('silent',bool),
    ('lang',str),
    ('ISATTY',bool),
]

ENV =Env(problem, service, args_list)
TAc =TALcolors(ENV)
LANG=Lang(ENV, TAc, lambda fstring: eval(f"f'{fstring}'"))

# START CODING YOUR SERVICE:
if not recognize(ENV['input_formula'], TAc, LANG, yield_feedback=False):
    ENV['silent'] = False

n_pairs = len(ENV['input_formula'])//2 
p = Par(n_pairs)
      
if ENV['silent'] and ENV['rank'] == p.rank(ENV['input_formula'],sorting_criterion=ENV['sorting_criterion']):
    exit(0)

TAc.print(LANG.opening_msg, "green")

recognize(ENV['input_formula'], TAc, LANG)

if ENV['rank'] < p.rank(ENV['input_formula'],sorting_criterion=ENV['sorting_criterion']):
    TAc.print(LANG.render_feedback("ranked-too-low", f"No. Your formula ranks higher than {ENV['rank']} among those with {n_pairs//2} pairs of parentheses (when sorted according to sorting_criterion={ENV['sorting_criterion']})."), "red", ["bold"])
    exit(0)
if ENV['rank'] > p.rank(ENV['input_formula'],sorting_criterion=ENV['sorting_criterion']):
    TAc.print(LANG.render_feedback("ranked-too-low", f"No. Your formula ranks lower than {ENV['rank']} among those with {n_pairs//2} pairs of parentheses (when sorted according to sorting_criterion={ENV['sorting_criterion']})."), "red", ["bold"])
    exit(0)
TAc.OK()
print(LANG.render_feedback("rank-ok", f"♥  You correctly ranked the formula among those on {n_pairs//2} pairs of parentheses (when sorted according to sorting_criterion={ENV['sorting_criterion']})."))
exit(0)