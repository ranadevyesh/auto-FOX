#!/usr/bin/env python
"""Entry points for Auto-FOX."""

import argparse
from os.path import isfile
from typing import Optional

from FOX import ARMC
from FOX.armc_functions.analyses import compare_pes_descriptors

__all__: list = []


def main_armc(args: Optional[list] = None) -> None:
    """Entrypoint for :meth:`FOX.classes.armc.ARMC.init_armc`."""
    parser = argparse.ArgumentParser(
         prog='FOX',
         usage='init_armc filename',
         description="Initalize the Auto-FOX Addaptive Rate Monte Carlo (ARMC) parameter optimizer.\
         See 'https://auto-fox.readthedocs.io/en/latest/4_monte_carlo.html' for a more detailed \
         description."
    )

    parser.add_argument(
        'filename', nargs=1, type=str, help='A .yaml file with ARMC settings'
    )

    filename = parser.parse_args(args).filename[0]
    if not isfile(filename):
        raise FileNotFoundError("[Errno 2] No such file: '{}'".format(filename))

    armc = ARMC.from_yaml(filename)
    armc.init_armc()


def main_plot_pes(args: Optional[list] = None) -> None:
    """Entrypoint for :func:`FOX.armc_functions.analyses.compare_pes_descriptors`."""
    parser = argparse.ArgumentParser(
         prog='FOX',
         usage='plot_pes input output -dset dset1 dset2 ...',
         description=""
    )

    parser.add_argument(
        'input', nargs=1, type=str, metavar='input',
        help='The path+name of the ARMC .hdf5 file'
    )

    parser.add_argument(
        '-o', '--output', nargs=1, type=str, metavar='output', required=False, default=[None],
        help=('Optional: The path+name of the to-be created .png file'
              'Set to the current working directory + the PES descriptor name by default')
    )

    parser.add_argument(
        '-i', '--iteration', nargs=1, type=int, default=[-1], required=False, metavar='iter',
        help=('Optional: The ARMC iteration containing the PES descriptor of interest.'
              'Set to the last iteration by default')
    )

    parser.add_argument(
        '-dset', '--datasets', nargs='+', type=str, metavar='datasets', required=True,
        dest='datasets', help='One or more names of hdf5 datasets containing PES descriptors.'
    )

    # Unpack arguments
    args_parsed = parser.parse_args(args)
    input_ = args_parsed.input[0]
    output = args_parsed.output[0]
    iteration = args_parsed.iteration[0]
    datasets = args_parsed.datasets

    if output is None:
        for dset in datasets:
            compare_pes_descriptors(input_, dset + '.png', dset, iteration=iteration)

    else:
        for i, dset in enumerate(datasets):
            compare_pes_descriptors(input_, str(i) + '_' + output, dset, iteration=iteration)
