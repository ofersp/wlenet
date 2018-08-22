#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from __future__ import print_function

import os
import sys
import subprocess
import argcomplete, argparse
import json
import pandas as pd

from scipy.io import savemat
from wlenet.misc.logger import Logger
from wlenet.misc.argconf import print_summary
from wlenet.misc.struct import Struct


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parallelize galsim_simulation.py via multi-processing')
    parser.add_argument('json_sim', type=str)
    parser.add_argument('phase', type=str, choices=['map', 'reduce'])
    parser.add_argument('test_train', type=str, choices=['test', 'train'])
    parser.add_argument('-p', '--spawn', type=str, choices=['subprocess', 'print'], default='print')
    parser.add_argument('-e', '--num_test_processes', type=int, default=-1)
    parser.add_argument('-r', '--num_train_processes', type=int, default=-1)
    parser.add_argument('-s', '--num_samples_per_process', type=int, default=-1)
    parser.add_argument('-o', '--out_base_path', type=str, default='')
    parser.add_argument('-j', '--json_map_reduce', type=str, default='')
    parser.add_argument('-l', '--logfile_stderr', type=str, default=None)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    if not args.json_map_reduce == '':
        with open(args.json_map_reduce) as json_fid:
            json_dict = json.load(json_fid)
        args.__dict__.update(json_dict)

    with open(args.json_sim) as json_sim_fid:
        json_sim_dict = json.load(json_sim_fid)
        args_sim = Struct()
        args_sim.__dict__.update(json_sim_dict)

    if args.logfile_stderr:
        sys.stderr = Logger(os.path.expanduser(args.logfile_stderr), sys.stderr)

    assert(not args.json_sim == '')
    script_dir, script_file = os.path.split(os.path.abspath(os.path.realpath(sys.argv[0])))
    python_exec = sys.executable

    print_summary(args, file=sys.stderr)
    print_summary(args_sim, file=sys.stderr, comment='args_sim')

    test_seeds_dynamic = range(1000, 1000 + args.num_test_processes)
    train_seeds_dynamic = range(1000, 1000 + args.num_train_processes)
    set_seeds = test_seeds_dynamic if args.test_train == 'test' else train_seeds_dynamic

    if args.phase == 'map':
        for seed_dynamic in set_seeds:
            cmd = [python_exec, '-u', script_dir + '/galsim_simulation.py',
                   '--json_config_path', args.json_sim,
                   '--test_train', args.test_train,
                   '--out_base_path', args.out_base_path,
                   '--seed_dynamic', str(seed_dynamic),
                   '--num_gals', str(args.num_samples_per_process)]
            if args.spawn == 'print':
                print(' '.join(cmd))
            else:
                subprocess.call(cmd)

    else: # reduce

        path_samples_out = os.path.expanduser(args.out_base_path + '/' + args.test_train + '_samples.bin')
        path_labels_out = os.path.expanduser(args.out_base_path + '/' + args.test_train + '_labels.bin')
        path_metadata_out = os.path.expanduser(args.out_base_path + '/' + args.test_train + '_metadata.pkl')

        assert(not os.path.exists(path_samples_out))
        f_samples_out = open(path_samples_out, 'wb')
        assert(not os.path.exists(path_labels_out))
        f_labels_out = open(path_labels_out, 'wb')
        assert(not os.path.exists(path_metadata_out))        
        full_df = None

        for seed_dynamic in set_seeds:

            path_samples_inp = os.path.expanduser('%s/%s_%07d_samples.bin' % (args.out_base_path, args.test_train, seed_dynamic))
            path_labels_inp = os.path.expanduser('%s/%s_%07d_labels.bin' % (args.out_base_path, args.test_train, seed_dynamic))
            path_metadata_inp = os.path.expanduser('%s/%s_%07d_metadata.pkl' % (args.out_base_path, args.test_train, seed_dynamic))

            f_samples_inp = open(path_samples_inp, 'rb')
            f_labels_inp = open(path_labels_inp, 'rb')            
            f_samples_out.write(f_samples_inp.read())
            f_labels_out.write(f_labels_inp.read())

            curr_df = pd.read_pickle(path_metadata_inp)
            if full_df is None:
                full_df = curr_df
            else:
                full_df = full_df.append(curr_df, ignore_index=True)

            f_samples_inp.close()
            f_labels_inp.close()

            os.remove(path_samples_inp)
            os.remove(path_labels_inp)
            os.remove(path_metadata_inp)

        f_samples_out.close()
        f_labels_out.close()
        full_df.to_pickle(path_metadata_out, compression=None)

        path_header = args.out_base_path + '/' + args.test_train + '_header.mat'
        path_header = os.path.expanduser(path_header)
        header = dict(sample_class='float32', sample_dim=args_sim.stamp_sz**2, sample_scale=1.0)
        savemat(path_header, header)
