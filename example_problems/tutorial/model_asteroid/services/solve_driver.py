#!/usr/bin/env python3
from sys import stderr

from multilanguage import Env, Lang, TALcolors
from TALinputs import TALinput
from TALfiles import TALfilesHelper

from math_modeling import ModellingProblemHelper

import asteroid_lib as al

# METADATA OF THIS TAL_SERVICE:
args_list = [
    ('source',str),
    ('m',int),
    ('n',int),
    ('instance_id',int),
    ('instance_format',str),
    ('sol_format',str), #(only_val|seq|subset|rows_columns)
    ('download',bool),
]

ENV =Env(args_list)
TAc =TALcolors(ENV)
LANG=Lang(ENV, TAc, lambda fstring: eval(f"f'{fstring}'"), print_opening_msg = 'now')
TALf = TALfilesHelper(TAc, ENV)


# START CODING YOUR SERVICE:
extension=al.format_name_to_file_extension(ENV['instance_format'], 'instance')
if TALf.exists_input_file('instance'):
    instance = al.get_instance_from_str(TALf.input_file_as_str('instance'), instance_format=ENV["instance_format"])
    TAc.print(LANG.render_feedback("instance-successfully-loaded", 'The file you have associated to `instance` filehandler has been successfully loaded.'), "yellow", ["bold"])
    # print(instance)
elif ENV["source"] == 'terminal':
    TAc.print(LANG.render_feedback("waiting", f'#? waiting for the {ENV["m"]} lines of {ENV["n"]} elements (0 or 1).\nFormat: you have to enter the {ENV["m"]} lines (corresponding to the {ENV["m"]} rows of the Asteroid matrix), where each of the {ENV["n"]} elements (0 or 1) must be separated by a space.\nAny line beggining with the "#" character is ignored.\nIf you prefer, you can use the "TA_send_txt_file.py" util here to send us the raw_instance of a file. Just plug in the util at the "rtal connect" command like you do with any other bot and let the util feed in the file for you rather than acting by copy and paste yourself.'), "yellow")
    instance = []
    for row in range (ENV['m']):
        TAc.print(LANG.render_feedback("new-row", f'Enter the row {row+1} of your Asteroid matrix (given by {ENV["n"]} elements (0 or 1) separated by a space):'), "yellow", ["bold"])
        instance.append([int(e) for e in TALinput(str, regex=f"^(([a-zA-Z0-9])*)$", sep=' ', TAc=TAc)])
elif ENV["source"] != 'catalogue':
    # Get random instance
    if ENV["source"] == 'random':
        instance = al.gen_instance(ENV['m'], ENV['n'], ENV['seed'])
        instance_str = al.instance_to_str(instance, instance_format=ENV['instance_format'])
        output_filename = f"instance_{ENV['m']}_{ENV['n']}_{ENV['seed']}.{ENV['sol_format']}.txt"
        TAc.print(LANG.render_feedback("instance-generation-successful", f'The instance has been successfully generated by the pseudo-random generator {ENV["source"]} called with arguments:\n   seed={ENV["seed"]}\n   m={ENV["m"]}\n   n={ENV["n"]}'), "yellow", ["bold"])
    else:
        assert False
else: # take instance from catalogue
    # Initialize ModellingProblemHelper
    mph = ModellingProblemHelper(TAc, ENV.INPUT_FILES, ENV.META_DIR)
    instance_str = TALf.get_catalogue_instancefile_as_str_from_id_and_ext(ENV["instance_id"], format_extension=al.format_name_to_file_extension(ENV["instance_format"],'instance'))
    instance = al.get_instance_from_str(instance_str, instance_format=ENV["instance_format"])
    TAc.print(LANG.render_feedback("instance-from-catalogue-successful", f'The instance with instance_id={ENV["instance_id"]} has been successfully retrieved from the catalogue.'), "yellow", ["bold"])
    output_filename = f"instance_catalogue_{ENV['instance_id']}.{ENV['sol_format']}.txt"

m=len(instance)
n=len(instance[0])
TAc.print(LANG.render_feedback("this-is-the-instance", 'This is the instance:'), "white", ["bold"])
TAc.print(al.instance_to_str(instance, instance_format=ENV['instance_format']), "white", ["bold"])

if ENV['source']=='random' and not TALf.exists_input_file('instance'):
    matrix=al.gen_instance(ENV['m'],ENV['n'],ENV['seed'])
else:
    matrix=instance
# print(matrix,m,n)
if ENV["sol_format"] == 'only_val':
    opt_val=al.opt_val(m,n,matrix)
    TAc.print(LANG.render_feedback("solution-val-title", f"The number of laser beams you need to shoot is:"), "green", ["bold"])
    TAc.print(LANG.render_feedback("solution-only_val", f'{opt_val}'), "white", ["reverse"])
    if ENV["download"]:
        TALf.str2output_file(opt_val,f'opt_val.txt')
else:
    TAc.print(LANG.render_feedback("solution-title", f"An optimal solution to this instance is:"), "green", ["bold"])
    # print('max_match_bip: ', al.max_match_bip(m,n,matrix))
    # print('max_match: ', al.max_match(m,n,matrix))
    solution=[elem for elem in al.min_cover(m,n,matrix)]
    if ENV["sol_format"] == 'seq':
        TAc.print(LANG.render_feedback("solution-seq", ' '.join(solution)), "white", ["reverse"])
        if ENV["download"]:
            TALf.str2output_file(' '.join(solution),output_filename)
    elif ENV["sol_format"] == 'subset':
        # print(solution)
        TAc.print(LANG.render_feedback("solution", al.subset_to_str(al.seq_to_subset(solution,m,n))), "white", ["reverse"])
        if ENV["download"]:
            TALf.str2output_file(al.subset_to_str(al.seq_to_subset(solution,m,n)),output_filename)
    elif ENV["sol_format"] == 'rows_columns':
        TAc.print(LANG.render_feedback("solution", al.sol_to_subset(solution,matrix)), "white", ["reverse"])
        if ENV["download"]:
            TALf.str2output_file(al.sol_to_subset(solution,matrix),output_filename)

exit(0)
