import scipy.io as sc
from scipy import signal as sig
from scipy import fft as fft
from scipy import fftpack as fftp
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import constants as cst


def data_1_verwerken(file):
    mat = sc.loadmat(file)
    x = np.array(mat.get('H'))
    return x

#def channel2APDP(PUNT):
   # pdp = np.zeros((METINGEN, TONEN))
  #  apdp = np.zeros((TONEN))
 #   frequentiekarakteristiek = np.zeros((METINGEN, TONEN), dtype=complex)
#    for i in range(METINGEN):
      #  for j in range(TONEN):
     #       frequentiekarakteristiek[i][j] = frequenties[j][PUNT][i]
    #for k in range(METINGEN):
   #     pdp[k] = abs(np.real((fft.ifft(frequentiekarakteristiek[k])))) #frequentiekarakteristiek omzetten naar PDP
  #      if (venster):
 #           pdp[k] = pdp[k]*sig.gaussian(TONEN, 100)
#    for TOON in range(TONEN):
       # som = 0
      #  for METING in range(METINGEN):
     #       som += pdp[METING][TOON]
    #    apdp[TOON] = 20*np.log10((som/METINGEN)*10**3)       #APDP
   # if (0):
  #      plt.plot(xas,apdp)
 #       plt.show()
#    return apdp


def powerdp(array_frequenties):
    if (venster):
        array_frequenties = array_frequenties * np.hanning(TONEN)  # windowing van frequentiekarakteristiek
    inverse = np.fft.ifft(array_frequenties)  # neem inverse ft van frequentiekarakteristiek
    pdp = np.sqrt((np.abs(inverse) ** 2) / TONEN)  # freq_kar omzetten in PDP (RMS)
    return pdp

def channel2APDP(PUNT):
    freq_kar = np.zeros(TONEN, dtype=complex)
    apdp = np.zeros(TONEN, dtype=complex)
    for meting in range(METINGEN):
        for toon in range(TONEN):
            freq_kar[toon] = frequenties[toon][PUNT][meting]
        pdp = 20 * np.log10(powerdp(freq_kar))
        for i in range(TONEN):  #gemiddelde van pdp's berekenen
            apdp[i] += pdp[i]
    apdp /= METINGEN
    if (PUNT == 0):
        plt.plot(apdp)
        plt.show()
    return apdp

def APDP2delays(APDP):
    APDP[APDP<offset] = offset #-100 voor dataset 1 -120 voor dataset2
    T = sig.argrelmax(APDP)
    if (0):
        plt.plot(APDP)
        plt.show()
    return T[0][0]*stap,T[0][1]*stap

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
    pass


def x_waarden():
    x = []
    for i in range(0, PUNTEN):
        theta = i * 15
        xn = 4 + 3 * math.sin(theta)
        x.append(xn)
    return x

def y_waarden():
    y = []
    for i in range(0, PUNTEN):
        theta = i * 15
        yn = 4 + 3 * math.sin(theta) * math.cos(theta)
        y.append(yn)
    return y




#Main
venster = 1

dataset = 1

if (dataset == 1):
    frequenties = data_1_verwerken('Dataset_1.mat')
    # stap = 1/2/(10**9)*200/201
    stap = 1 / 3 / (10 ** 9) * 200 / 201
    offset = -100
else:
    frequenties = data_1_verwerken('Dataset_2.mat')
    stap = 1 / 11 / 10 ** 9 * 999 / 1000
    offset = -120

TONEN = len(frequenties)
PUNTEN = len(frequenties[0])
METINGEN = len(frequenties[0][0])

tau = delays_berekenen()

x_coordinaat = []
y_coordinaat = []

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