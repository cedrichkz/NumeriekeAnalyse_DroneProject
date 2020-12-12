import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from scipy import signal
import scipy.io as sc
from math import sqrt
from time import sleep


def data_1_verwerken(file):
    mat = sc.loadmat(file)
    arr = np.array(mat.get('H')) #geeft 3D array terug in vorm [METING][PUNTEN][TONEN]
    return arr

def channel2APDP(array_frequenties, plotten = False):
    #venster over freq_kar
    array_frequenties = array_frequenties * np.hanning(200) #windowing van frequentiekarakteristiek
    inverse = np.fft.ifft(array_frequenties)                #neem inverse ft van frequentiekarakteristiek
    n = len(array_frequenties)
    pdp =  np.sqrt((np.abs(inverse)**2)/n)                  #freq_kar omzetten in PDP (RMS)

    if(plotten):
        #PDP plotten
        fig = plt.figure()
        plt.title("PDP")
        plt.xlabel("tijd")
        plt.ylabel("vermogen")
        plt.plot(pdp)

        #Automatisch de plot sluiten na 1 sec
        plt.show(block=False)
        plt.pause(1)
        plt.close(fig)


#PDP tonen van de eerste metingen voor de 24 punten
gegevens = data_1_verwerken("Dataset_1")
AANTAL_Freq = 200
AANTAL_Punten = 24
for j in range(AANTAL_Punten):
    ArrFreq = []
    for i in range(AANTAL_Freq):
        ArrFreq.append(gegevens[i][j][0])
    channel2APDP(ArrFreq, True)


    



