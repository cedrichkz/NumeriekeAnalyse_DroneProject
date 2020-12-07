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

def channel2APDP(PUNT):
    pdp = np.zeros((METINGEN, TONEN))
    apdp = np.zeros((TONEN))
    frequentiekarakteristiek = np.zeros((METINGEN, TONEN), dtype=complex)
    for i in range(METINGEN):
        for j in range(TONEN):
            frequentiekarakteristiek[i][j] = frequenties[j][PUNT][i]
    for k in range(METINGEN):
        pdp[k] = abs(((fft.ifft([k])))) #frequentiekarakteristiek omzetten naar PDP

    for TOON in range(TONEN):
        som = 0
        for METING in range(METINGEN):
            som += pdp[METING][TOON]
        apdp[TOON] = 20*np.log10((som/METINGEN))       #APDP
    if(1):  #APDP te plotton op 1
        plt.plot(apdp)
        plt.show()
    return apdp


def APDP2delays(APDP):
    #T0 = np.argmax(APDP)
    #lage_index = 0
    #for i in range(T0+1,len(APDP)-1):
    #    if APDP[i]<APDP[i+1]:
    #        lage_index = i
    #        print(lage_index)
    #        break
    #T1 = np.argmax(APDP[lage_index:len(APDP)-10]) + i
    #print(T0,T1)
    #return T0*stapA, T1*stapA
    peaks = sig.find_peaks(APDP, prominence=1) #functie om toppen te vinden
    print(peaks[0])
    T0 = peaks[0][0]*stapA                      #index vermenigvuldigen met stap
    if (len(peaks[0]) >1):
        T1 = peaks[0][1]*stapA
    else:
        T1 = (peaks[0][0] + 10) * stapA
    return T0,T1

def delays_berekenen():
    delays = np.zeros((PUNTEN,2))
    for PUNT in range(PUNTEN):
        APDP = channel2APDP(PUNT)
        delays[PUNT] = APDP2delays(APDP)
    return delays

#def snelheid_berekenen(x, y, tau):
 #   return math.sqrt(x**2+y**2)/tau

def calculate_location(tau0, tau1, v):
    y = ((tau1**2-tau0**2)*v**2)/4
    x = (tau0*v)**2-(y-1)**2

    return x,y


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


frequenties = data_1_verwerken('Dataset_1.mat')
METINGEN = len(frequenties[0][0])
TONEN = len(frequenties)
PUNTEN = len(frequenties[0])
#print(PUNTEN, TONEN, METINGEN)
#frequenties[TONEN][PUNTEN][METINGEN]

stapA = 1/3/(10**9)*200/201
stapB = 1/11/10**9*999/1000
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
plt.scatter(x_juist, y_juist)



plt.show()