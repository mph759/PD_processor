"""
Reads and parses .ref files from Jana to find the solution with the lowest wRp and
outputs the relevant parameters to a csv file
Author: Michael Hassett
Created: 2024-05-14
"""

from pprint import pprint
import re
import pandas as pd
from pathlib import Path


def get_fitting_params(ref_file: Path):
    with open(ref_file, 'r') as f:
        for line in f:
            if line.startswith('|GOF'):
                results = line.strip('|\s+\n')
                results = results.split(' ')
                results = list(filter(None, results))
                results = [x for x in results if x != '=']
                for i in range(1, len(results), 2):
                    results[i] = float(results[i])
                res_dct = {results[i]: results[i + 1] for i in range(0, len(results), 2)}
                return res_dct


def get_unit_cell_params(ref_file: Path):
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
    cell_params.rename(index=dict(zip(cell_params.index,['value', 'error'])), inplace=True)

    return cell_params


def get_space_group(ref_file: Path):
    with open(ref_file, 'r') as f:
        for line in f:
            if line.startswith('Centrosymmetric space group'):
                space_group = line.strip('\n')
                space_group = space_group.split(' ')[3:]
                space_group = (space_group[0], int(space_group[-1]))
                return space_group


if __name__ == '__main__':
    base_dir = Path(
        r'C:\Users\Michael_X13\OneDrive - RMIT University\Research\2021 - Honours Research Project\Data '
        r'Analysis\Jack\pil-xrd\Binns_17649')
    samples = ['AMAMS', 'AMAN', 'BAMS', 'EATFA/EATFA/ht_refinement']
    for sample in samples:
        current_best = None
        sample_folder = base_dir / sample
        ref_files = sample_folder.glob('*.ref')
        results = {}
        for ref_file in ref_files:
            results[ref_file.name] = get_fitting_params(ref_file)
            if results[ref_file.name] is None:
                continue
            if current_best is None:
                current_best = ref_file.name
            elif results[ref_file.name]['wRp'] < results[current_best]['wRp']:
                current_best = ref_file.name
        best_file = sample_folder / current_best
        print(f"Best wRp for {sample} is {results[current_best]['wRp']} from {current_best}")

        unit_cell_params = get_unit_cell_params(best_file)
        space_group = get_space_group(best_file)

        print(space_group)
        print(unit_cell_params)
        print('\n')

