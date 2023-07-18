# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 13:48:00 2021

@author: Michael Hassett

Parsing code to get temperature values from .parab files
Files from Australian Synchrotron, ANSTO, PD Beamline (23-03-2021)

Last edited on Tue Jun 8 11:06:00 2021
"""

import glob
from pathlib import Path
import numpy as np


def TempExtract(folder_dir):
    temps = []
    for file in glob.glob(str(Path.cwd())+folder_dir+"/*.parab"):
            f = open(file)
            file_name = (file.split("\\")[-1]).strip(".parab")
            index = str(int(file_name.split("_")[-1])).zfill(4)
            p = file_name.split("_")[-2]
            
            for line in f.readlines():
                if 'Cryostream_Temperature_(K)' in line:
                    temperature = line.split()[1]            
                    temps.append([p,index,temperature])
    temps = np.array(temps)
    return temps

def TempAverage(folder_dir):
    temps = TempExtract(folder_dir)
    p1 = []
    p2 = []
    p12 = []
    for i in temps:
        if i[0]=="p1":
            p1.append([i[1],i[2]])
        else:
            p2.append([i[1],i[2]])
    p1 = np.array(p1)
    p2 = np.array(p2)
    if len(p1)>=len(p2):
        for temp1 in p1:
            for temp2 in p2:
                if temp1[0] == temp2[0]:
                    p12.append([temp1[0],np.average([float(temp1[1]),float(temp2[1])]),np.std([float(temp1[1]),float(temp2[1])])])
                    break
    else:
        for temp2 in p2:
            for temp1 in p1:
                if temp1[0] == temp2[0]:
                    p12.append([temp1[0],np.average([float(temp1[1]),float(temp2[1])]),np.std([float(temp1[1]),float(temp2[1])])])
                    break
    p12 = np.array(p12)
    return p12

def WriteFile(folder_dir):
    csv_name = str(folder_dir.split("/")[2])+"_"+str(folder_dir.split("/")[3])+'.csv'
    print(csv_name)
    temps = TempAverage(folder_dir)
    print(temps)
    np.savetxt(csv_name,temps,fmt=["%s","%s","%s"],delimiter=",")

if __name__ == "__main__":    
    folder_directory = "/EAA/EAA/VT"
    WriteFile(folder_directory)