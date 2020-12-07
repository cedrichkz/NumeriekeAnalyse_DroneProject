import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from scipy import signal
import scipy.io as sc


def data_1_verwerken(file):
    mat = sc.loadmat(file)
    arr = np.array(mat.get('H')) #geeft 3D array terug in vorm [METING][PUNTEN][TONEN]
    return arr

def channel2APDP(array_frequenties, plotten = False):
    #venster over freq_kar
    inverse = np.fft.ifft(array_frequenties)    #neem inverse ft van frequentiekarakteristiek
    pdp = np.abs(inverse)**2

    if(plotten):
        plt.title("Inverse FT")
        plt.xlabel("tijd")
        plt.ylabel("Amplitude")
        plt.plot(inverse)

        plt.figure()
        plt.title("PDP")
        plt.xlabel("tijd")
        plt.ylabel("vermogen")
        plt.plot(pdp)
        plt.show()

gegevens = data_1_verwerken("Dataset_1")

ArrFreq = []
AANTAL_Freq = 200
for i in range(AANTAL_Freq):
    ArrFreq.append(gegevens[i][0][0])

channel2APDP(ArrFreq, True)

    



