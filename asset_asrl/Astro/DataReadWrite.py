import numpy as np
import os
import csv
import asset_asrl.Astro.Constants as c
norm = np.linalg.norm


def WriteData(traj,name,Folder = 'Data'):
    if(Folder != ''):
        if not os.path.exists(Folder+'/'):
            os.makedirs(Folder+'/')
    np.save(Folder+'/' + name+ '.npy',traj)
   
def ReadData(name,Folder = 'Data'):
   return np.load(Folder+'/' + name+ '.npy',allow_pickle=True)

def ReadCopernicusFile(name,Folder = 'Data'):
    StateHist = []
    with open(Folder+'/' + name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for i,row in enumerate(readCSV):
            if(i>4):
                S=np.zeros((8))
                for j in range(0,6):
                    S[j] = float(row[4+j])*1000.0
                S[6]=float(row[3])*c.day
                S[7]=float(row[1])
                StateHist.append(S)
    return StateHist
           


