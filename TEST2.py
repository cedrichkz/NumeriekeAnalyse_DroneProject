import scipy.io as sc
from scipy import signal as sig
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import constants as cst


def data_1_verwerken(bestand):
    mat = sc.loadmat(bestand)
    gegevens = np.array(mat.get('H'))
    return gegevens


def powerdp(array_frequenties):
    if (venster):
        array_frequenties = array_frequenties * np.hanning(AANTAL_TONEN)  # windowing van frequentiekarakteristiek
    inverse = np.fft.ifft(array_frequenties)  # neem inverse ft van frequentiekarakteristiek
    pdp = np.sqrt((np.abs(inverse) ** 2) / AANTAL_TONEN)  # freq_kar omzetten in PDP (RMS)
    return pdp


def channel2APDP(PUNT):
    freq_kar = np.zeros(AANTAL_TONEN, dtype=complex)  # lege array voor de freq_kar in te plaatsen
    apdp = np.zeros(AANTAL_TONEN, dtype=complex)  # lege array waarmee adpd zal berekent worden

    for meting in range(AANTAL_METINGEN):
        for toon in range(AANTAL_TONEN):  # de freq_kar in een array steken
            freq_kar[toon] = frequenties[toon][PUNT][meting]

        pdp = 20 * np.log10(powerdp(freq_kar))  # de pdp schalen voor de gemiddelde te nemen
        for i in range(AANTAL_TONEN):  # de geschaalde pdp's in een nieuwe array steken
            apdp[i] += pdp[i]

    apdp /= AANTAL_METINGEN  # gemiddelde van pdp's berekenen
    if (PUNT == 0):
        plt.plot(apdp)
        plt.show()
    return apdp


def APDP2delays(APDP, plotten=False):
    APDP[APDP < offset] = offset  # -100 voor dataset 1 -120 voor dataset2
    T = sig.argrelmax(APDP)  # pieken van ADPD bepalen
    if (plotten):
        plt.plot(APDP)
        plt.show()
    return T[0][0] * stap, T[0][1] * stap


# voor elk punt tau0 en tau1 bepalen
def delays_berekenen():
    delays = np.zeros((AANTAL_PUNTEN, 2))
    for PUNT in range(AANTAL_PUNTEN):
        APDP = channel2APDP(PUNT)
        delays[PUNT] = APDP2delays(APDP)
    return delays


def calculate_location(tau0, tau1, v):
    y = ((tau1 ** 2 - tau0 ** 2) * v ** 2) / 4
    x = (tau0 * v) ** 2 - (y - 1) ** 2
    return x, y


def fout_berekenen():
    pass


# de correcte x-waarden bijhouden
def x_waarden():
    x = []
    for i in range(0, AANTAL_PUNTEN):
        theta = i * 15
        xn = 4 + 3 * math.sin(theta)
        x.append(xn)
    return x

# de correcte y-waarden bijhouden
def y_waarden():
    y = []
    for i in range(0, AANTAL_PUNTEN):
        theta = i * 15
        yn = 4 + 3 * math.sin(theta) * math.cos(theta)
        y.append(yn)
    return y

# MAIN

dataset = 2
venster = 1

if (dataset == 1):
    frequenties = data_1_verwerken('Dataset_1.mat')
    stap = 1/2/(10**9)*200/201
    offset = -100
else:
    frequenties = data_1_verwerken('Dataset_2.mat')
    print()
    stap = 1 / 11 / 10 ** 9 * 999 / 1000
    offset = -120

AANTAL_TONEN = len(frequenties)
AANTAL_PUNTEN = len(frequenties[0])
AANTAL_METINGEN = len(frequenties[0][0])

tau = delays_berekenen()  # tau0 en tau1 bijhouden van elk punt

x_berekend = []
y_berekend = []

x_juist = x_waarden()
y_juist = y_waarden()

# coordinaten bepalen van elk punt
for i in range(0, AANTAL_PUNTEN):
    coord = calculate_location(tau[i][0], tau[i][1], cst.speed_of_light)
    print('Punt', i, ':', coord, '\nFout op x en y= ', x_juist[i] - coord[0], " ; ", y_juist[i] - coord[0])
    x_berekend.append(coord[0])
    y_berekend.append(coord[1])


plt.xlabel("x")
plt.ylabel("y")
plt.plot(x_berekend, y_berekend)
plt.scatter(x_berekend, y_berekend)
#plt.scatter(x_juist, y_juist)

plt.show()
