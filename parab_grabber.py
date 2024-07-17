# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 13:48:00 2021

@author: Michael Hassett

Parsing code to get temperature values from .parab files
Files from Australian Synchrotron, ANSTO, PD Beamline (23-03-2021)

Last edited on Tue Jun 8 11:06:00 2021
"""

from pathlib import Path
import numpy as np
import pandas as pd
from uncertainties import ufloat


def temp_extract(folder_dir: Path):
    temps = []
    for file in folder_dir.glob(f"**/*.parab"):
        f = open(file)
        file_name = file.stem
        index = str(int(file_name.split("_")[-1])).zfill(4)
        p = file_name.split("_")[-2]

        for line in f.readlines():
            if 'Cryostream_Temperature_(K)' in line:
                temperature = line.split()[1]
                temps.append([p, index, temperature])
    temps = np.array(temps)
    return temps


def temp_average(folder_dir: Path):
    temps = temp_extract(folder_dir)
    p1 = []
    p2 = []
    p12 = []
    for i in temps:
        if i[0] == "p1":
            p1.append([i[1], i[2]])
        else:
            p2.append([i[1], i[2]])
    p1 = np.array(p1)
    p2 = np.array(p2)
    if len(p1) >= len(p2):
        for temp1 in p1:
            for temp2 in p2:
                if temp1[0] == temp2[0]:
                    value = np.average([float(temp1[1]), float(temp2[1])])
                    uncertainty = np.std([float(temp1[1]), float(temp2[1])])
                    value_string = ufloat(value, uncertainty)
                    p12.append([temp1[0], value_string])
                    break
    else:
        for temp2 in p2:
            for temp1 in p1:
                if temp1[0] == temp2[0]:
                    p12.append([temp1[0], np.average([float(temp1[1]), float(temp2[1])]),
                                np.std([float(temp1[1]), float(temp2[1])])])
                    break
    p12 = np.array(p12)
    return p12

def temp_range(df, sample):
    temp_value = df['temperature']
    print(temp_value)
    EAN = {'name': 'EAN', 'alpha': (123.8, 210), 'beta': (214, 278)}
    EATFA = {'name': 'EATFA', 'alpha': (124, 192), 'beta': (178, 260)}
    DEAF = {'name': 'DEAF', 'alpha': (124, 300),  'beta': (305, 999)}
    temp_ranges = pd.DataFrame([EAN, EATFA, DEAF], columns=['name', 'alpha', 'beta']).set_index('name')

    temp_range = temp_ranges.loc[sample]

    if sample in temp_ranges.index:
        if temp_value >= temp_range['alpha'][0] and df['temperature'] <= temp_range['alpha'][1]:
            return 'alpha'
        elif temp_value >= temp_range['beta'][0] and df['temperature'] <= temp_range['beta'][1]:
            return 'beta'
        else:
            return None
    else:
        raise ValueError('Sample value not in range')

def write_file(folder_dir):
    csv_name = str(folder_dir.split("/")[2]) + "_" + str(folder_dir.split("/")[3]) + '.csv'
    print(csv_name)
    temps = temp_average(folder_dir)
    print(temps)
    np.savetxt(csv_name, temps, fmt=["%s", "%s", "%s"], delimiter=",")


def main():
    data_dir = Path(
        r'C:\Users\Michael_X13\OneDrive - RMIT University\Research\2021 - Honours Research Project\Synchrotron Beamtime\Data')
    sample = 'DEAF'
    sample_dir = r'DEAF\DEAFwool2\VT'
    temps = temp_average(data_dir / sample_dir)

    df = pd.DataFrame(temps, columns=['index', 'temperature']).set_index('index')
    df['temp_string'] = df['temperature'].apply(lambda x: f'{x:0.1uS}')
    df['phase'] = df.apply(temp_range, sample=sample, axis=1)
    print(df['temperature'])
    df.to_csv(Path(Path.cwd() / 'output' / 'temps.csv'))


if __name__ == "__main__":
    main()
