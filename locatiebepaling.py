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
    freq_kar = np.zeros(FREQUENTIES, dtype=complex)
    apdp = np.zeros(FREQUENTIES, dtype=complex)
    for meting in range(METINGEN):
        for freq in range(FREQUENTIES):
            freq_kar[freq] = data[freq][PUNT][meting]
        pdp = 20 * np.log10(powerdp(freq_kar))
        for i in range(FREQUENTIES):  # gemiddelde van pdp's berekenen
            apdp[i] += pdp[i]
    apdp /= METINGEN
    if PUNT == 100:
        plt.plot(apdp)
        plt.show()
    return apdp

def APDP2delays(APDP):
    APDP[APDP<offset] = offset          # ruis in vlakke stukken elimineren
    T = sig.argrelmax(APDP)             # indexen van de pieken bepalen
    return T[0][0]*stap,T[0][1]*stap    # index * tijd tussen 2 frequenties om tau te bepalen

def delays_berekenen():
    delays = np.zeros((PUNTEN,2))
    for PUNT in range(PUNTEN):
        APDP = channel2APDP(PUNT)
        delays[PUNT] = APDP2delays(APDP)
    return delays

#def calculate_location(tau0, tau1):
#    y = ((tau1**2-tau0**2)*cst.speed_of_light**2)/4
#    x = ((tau0*cst.speed_of_light)**2-(y-1)**2)

#    return x,y

def calculate_location(tau0, tau1):
    a = tau1  * cst.speed_of_light
    b = tau0  * cst.speed_of_light
    y = (a**2 - b**2)/4
    x = np.sqrt(-(a)**4 + 2 * a**2 * b**2 + 8*a**2 - b**4 + 8*b**2 - 16)/4
    return x, y



def fout_berekenen():
    fout_x = []
    fout_y = []
    fout = []
    for i in range(PUNTEN):
        fout_x.append(x_coordinaat[i] - x_juist[i])
        fout_y.append(y_coordinaat[i] - y_juist[i])
    for i in range(PUNTEN):
        fout.append(np.sqrt(fout_x[i]**2 + fout_y[i]**2))
    list.sort(fout)
    mediaan = (fout[11] + fout[12])/2
    print("fout = ", mediaan)


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

tau = delays_berekenen()

x_coordinaat = []
y_coordinaat = []

for i in range(0,PUNTEN):
    coord = calculate_location(tau[i][0],tau[i][1])
    print('Punt',i,':',coord)
    x_coordinaat.append(coord[0])
    y_coordinaat.append(coord[1])


x_voorlopig = x_waarden()
y_voorlopig = y_waarden()
volgorde = [0,13,8,21,3,16, 11, 6,19,1,14,9,22,4,17,12,7,20,2,15,10,23,5,18]
y_juist = []
x_juist = []
for i in volgorde:
    x_juist.append(x_voorlopig[i])
    y_juist.append(y_voorlopig[i])

nummers = []
for i in range(PUNTEN):
    nummers.append(i)

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
ax1.scatter(x_coordinaat, y_coordinaat)
ax2.scatter(x_juist, y_juist)
for i, txt in enumerate(nummers):
    ax1.annotate(txt, (x_coordinaat[i], y_coordinaat[i]))
    ax2.annotate(txt, (x_juist[i], y_juist[i]))



plt.show()

fout_berekenen()


plt.scatter(x_coordinaat, y_coordinaat)
plt.plot(x_coordinaat, y_coordinaat)
plt.scatter(x_juist, y_juist)
plt.plot(x_juist, y_juist)

plt.show()
