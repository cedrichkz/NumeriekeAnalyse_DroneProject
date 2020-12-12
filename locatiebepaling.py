import scipy.io as sc
from scipy import signal as sig
from scipy import fft as fft
from scipy import fftpack as fftp
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import constants as cst


def data_1_verwerken(file):
    mat = sc.loadmat(file)      # data inladen
    x = np.array(mat.get('H'))  # data omzetten naar een array
    return x

def powerdp(array_frequenties):
    if (venster):
        array_frequenties = array_frequenties * np.hanning(FREQUENTIES)  # windowing van frequentiekarakteristiek
    inverse = np.fft.ifft(array_frequenties)                             # neem inverse ft van frequentiekarakteristiek
    pdp = np.sqrt((np.abs(inverse) ** 2) / FREQUENTIES)                  # freq_kar omzetten in PDP (RMS)
    return pdp

def channel2APDP(PUNT):
    freq_kar = np.zeros(FREQUENTIES, dtype=complex) # lege array voor de freq_kar in te plaatsen
    apdp = np.zeros(FREQUENTIES, dtype=complex)     # lege array waarmee adpd zal berekent worden
    for meting in range(METINGEN):
        for freq in range(FREQUENTIES):                 # de freq_kar in een array steken
            freq_kar[freq] = data[freq][PUNT][meting]
        pdp = 20 * np.log10(powerdp(freq_kar))          # de pdp schalen voor de gemiddelde te nemen
        for i in range(FREQUENTIES):
            apdp[i] += pdp[i]
    apdp /= METINGEN    # gemiddelde van pdp's berekenen
    if PUNT == 100:
        plt.plot(apdp)
        plt.show()
    return apdp

def APDP2delays(APDP):
    APDP[APDP<offset] = offset          # ruis in vlakke stukken elimineren
    T = sig.argrelmax(APDP)             # indexen van de pieken bepalen
    return T[0][0]*stap,T[0][1]*stap    # index * tijd tussen 2 frequenties om tau te bepalen

# voor elk punt tau0 en tau1 bepalen
def delays_berekenen():
    delays = np.zeros((PUNTEN,2))
    for PUNT in range(PUNTEN):
        APDP = channel2APDP(PUNT)
        delays[PUNT] = APDP2delays(APDP)
    return delays

def calculate_location(tau0, tau1, v):
    y = ((tau1**2-tau0**2)*v**2)/4
    x = (tau0*v)**2-(y-1)**2
    return x,y


def fout_berekenen():
    fout_x = []
    fout_y = []
    fout = []
    for i in range(PUNTEN):
        fout_x.append(x_coordinaat[i] - x_juist[i])
        fout_y.append(y_coordinaat[i] - y_juist[i])
        fout.append(np.sqrt(fout_x[i]**2 + fout_y[i]**2))

    list.sort(fout)
    mediaan = (fout[11] + fout[12])/2
    print("fout = ", mediaan)

# de correcte x-waarden bijhouden
def x_waarden():
    x = []
    for i in range(0, PUNTEN):
        theta = i * 15
        xn = 4 + 3 * math.sin(theta)
        x.append(xn)
    return x

# de correcte y-waarden bijhouden
def y_waarden():
    y = []
    for i in range(0, PUNTEN):
        theta = i * 15
        yn = 4 + 3 * math.sin(theta) * math.cos(theta)
        y.append(yn)
    return y


#Main
venster = 1

dataset = 2

if (dataset == 1):
    data = data_1_verwerken('Dataset_1.mat')
    stap = 1/10/(10**6)/200
    offset = -100
else:
    data = data_1_verwerken('Dataset_2.mat')
    stap = 1/10/(10**6)/1000
    offset = -120

FREQUENTIES = len(data)
PUNTEN = len(data[0])
METINGEN = len(data[0][0])

tau = delays_berekenen() # tau0 en tau1 bijhouden van elk punt

x_coordinaat = []
y_coordinaat = []

# coordinaten bepalen van elk punt
for i in range(0,PUNTEN):
    coord = calculate_location(tau[i][0],tau[i][1], cst.speed_of_light)
    print('Punt',i,':',coord)
    x_coordinaat.append(coord[0])
    y_coordinaat.append(coord[1])


x_juist = x_waarden()
y_juist = y_waarden()

plt.scatter(x_coordinaat, y_coordinaat)
plt.plot(x_coordinaat, y_coordinaat)
plt.scatter(x_juist, y_juist)

plt.show()

fout_berekenen()