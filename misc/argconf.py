from __future__ import print_function
import sys
from os.path import split, abspath, realpath

def print_summary(args, file=sys.stdout, comment=None):

    script_name = split(abspath(realpath(sys.argv[0])))[1]
    if comment is None:
        comment = 'args'
    title = 'Configuration summary for %s (%s)' % (script_name, comment)
    print('', file=file)
    print(title, file=file)
    print('-'*len(title), file=file)
    sorted_args_dict_items = sorted(args.__dict__.items())
    for arg in sorted_args_dict_items:
        arg1 = arg[1] if arg[1] != None else 'None'
        print('{:<30} {:<60}'.format(arg[0], arg1), file=file)
    print('', file=file)
