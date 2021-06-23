#!/usr/bin/env python3

from sys import stderr, exit, argv

from multilanguage import Env, Lang, TALcolors
from TALinputs import TALinput

from zaino_lib import *

# METADATA OF THIS TAL_SERVICE:
problem="knapsack"
service="con2opt"
args_list = [
    ('size',str),
    ('seed',str),
    ('goal',str),
    ('lang',str),
]

ENV =Env(problem, service, args_list)
TAc =TALcolors(ENV)
LANG=Lang(ENV, TAc, lambda fstring: eval(f"f'{fstring}'"))
TAc.print(LANG.opening_msg, "green")

size = ENV['size']
seed = ENV['seed']

a, W, wt, val, n = GenZcon(size, seed)
TAc.print(f"\nSeed dell'istanza: {a}\n", "yellow")
TAc.print(f"{n} {W}", "yellow")
a_wt = wt.split(",")
a_val = val.split(",")
for x,y in zip(a_wt,a_val):
   TAc.print(x+" "+y,"yellow")
a_wt = [int(i) for i in a_wt]
a_val = [int(i) for i in a_val]
answer = zcon(W, a_wt, a_val, n)
print("\n")
prompt = input()
count = 0
num_space = prompt.count(" ")

while num_space!=n-1:
    ps_n, ps_W = prompt.split()
    ps_wt = ""
    ps_val = ""
    for i in range(int(ps_n)):
        if i == 0:
            PSwt, PSval = input().split()
            ps_wt = PSwt
            ps_val = PSval
        else:
            PSwt, PSval = input().split()
            ps_wt = ps_wt+","+PSwt
            ps_val = ps_val+","+PSval           
    a_ps_wt = ps_wt.split(",")
    a_ps_val = ps_val.split(",")
    a_ps_wt = [int(i) for i in a_ps_wt]
    a_ps_val = [int(i) for i in a_ps_val]
    o_zopt = zopt(int(ps_W), a_ps_wt, a_ps_val, int(ps_n))
    count += 1
    TAc.print(o_zopt, "yellow")
    prompt = input()
    num_space = prompt.count(" ")

if ENV['goal'] == "correct":
    if str(answer) == prompt:
            TAc.print("\nCORRETTO!", "green")
    else:
            TAc.print("\nSBAGLIATO!", "red")

if ENV['goal'] == "number_of_calls_linear_in_n":
    if str(answer) == prompt:
            if count <= n+1:
                TAc.print("\nCORRETTO! La tua soluzione è anche ottima, hai usato un numero lienare di chiamate a oracolo.", "green")
            if count > n+1:
                TAc.print("\nCORRETTO! Esiste una soluzione più efficiente però che usa un numero lienare di chiamate all'oracolo.", "yellow")
    else:
            TAc.print("\nSBAGLIATO!", "red")