"""
Reads and parses .ref files from Jana to find the solution with the lowest wRp and
outputs the relevant parameters to a csv file
Author: Michael Hassett
Created: 2024-05-14
"""

from pprint import pprint
import re
from typing import Tuple

import pandas as pd
import numpy as np
from pathlib import Path
from uncertainties import ufloat


def get_fitting_params(ref_file: Path) -> dict:
    with open(ref_file, 'r') as f:
        for line in f:
            if line.startswith('|GOF'):
                print('Found GOF')
                results = line.strip('|\s+\n')
                results = results.split(' ')
                results = list(filter(None, results))
                results = [x.strip('=') for x in results if x != '=']

                for i in range(1, len(results), 2):
                    results[i] = float(results[i])
                res_dct = {results[i]: results[i + 1] for i in range(0, len(results), 2)}
                return res_dct
    print('No GOF found')


def get_unit_cell_params(ref_file: Path) -> pd.DataFrame:
    if not ref_file:
        return None
    with open(ref_file, 'r') as f:
        last_line = len(f.readlines())
    header_line = last_line
    end_line = last_line
    with open(ref_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if line.startswith(
                    '               a         b         c       alpha     beta      gamma     Volume    Density'):
                header_line = i
            if i > header_line and line.startswith('='):
                end_line = i
                break
    cell_params = pd.read_csv(ref_file, sep='\s+', skiprows=end_line - 2, header=None,
                              names=['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'volume', 'density'], nrows=2)
    cell_params.rename(index=dict(zip(cell_params.index, ['value', 'error'])), inplace=True)
    full_cell_params = {column_name: (f'{ufloat(value, error):0.1uS}' if error else f'{int(value)}') for
                        column_name, (value, error) in cell_params.items()}
    return full_cell_params


def get_space_group(ref_file: Path) -> tuple[None, None] | tuple[str, int]:
    if not ref_file:
        return None, None
    with open(ref_file, 'r') as f:
        for line in f:
            if 'centrosymmetric space group' in line.lower():
                space_group = line.strip('\n')
                space_group = space_group.split(' ')[3:]
                space_group = (space_group[0], int(space_group[-1]))
                return space_group


def get_best_dir(data_folder) -> tuple:
    current_best = None
    ref_files = list(data_folder.glob('**/*.ref'))
    if len(list(ref_files)) == 0:
        print('No .ref files found')
        return None, None
    else:
        print(f'Found {len(list(ref_files))} refs for {data_folder}')
    results = {}
    for ref_file in ref_files:
        print(ref_file)
        results[ref_file] = get_fitting_params(ref_file)
        if results[ref_file] is None:
            continue
        if current_best is None:
            current_best = ref_file
        elif results[ref_file]['wRp'] < results[current_best]['wRp']:
            current_best = ref_file
    if current_best is None:
        return None, None
    print(results)
    best_file = current_best
    best_wRp = results[best_file]['wRp']
    return best_file, best_wRp


def get_all_data(data: pd.Series) -> pd.Series:
    data['file'], data['wRp'] = get_best_dir(data['file'] / data['sample'])
    if data['file'] is None:
        print(f'No data found in refs for {data["sample"]}')
        return data
    try:
        space_group = get_space_group(data['file'])
        data['space group'], data['space group no'] = space_group

        data.update(get_unit_cell_params(data['file']))
    except TypeError:
        print('No unit cell found')
        return data

    return data


if __name__ == '__main__':

    # base_dir = Path(r'C:\Users\Michael_X13\OneDrive - RMIT University\Research\2021 - Honours Research Project\Data Analysis\Jack\pil-xrd\Binns_17649')
    # samples = ['AMAMS', 'AMAN', 'BAMS', 'EATFA', 'BAA', 'BAN', 'EACl', 'EAMS', 'EAPC', 'EAPH', 'EtAMS', 'EtAPH']

    base_dir = Path(r'C:\Users\Michael_X13\OneDrive - RMIT University\Research\2021 - Honours Research Project\Data Analysis')
    # samples = ['DEAF/alpha', 'DEAF/beta', 'BAA', 'EAMS', 'EAN/alpha', 'EAN/beta', 'EATFA/alpha', 'EATFA/beta']
    samples = ['EAA', 'EACl']
    df = pd.DataFrame({'sample': samples, 'file': base_dir, 'wRp': np.nan,
                       'space group': None, 'space group no': np.nan,
                       'a': np.nan, 'b': np.nan, 'c': np.nan,
                       'alpha': np.nan, 'beta': np.nan, 'gamma': np.nan,
                       'volume': np.nan, 'density': np.nan})

    df = df.apply(get_all_data, axis=1, result_type='broadcast')

    output_dir = Path.cwd() / 'output'
    df.to_csv(output_dir / 'ref_data_v3.csv', index=False, encoding='utf-8')
