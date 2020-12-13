import scipy.io as sc
from scipy import signal as sig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from scipy import constants as cst


def data_verwerken(file):
    mat = sc.loadmat(file)  # data inladen
    x = np.array(mat.get('H'))  # data omzetten naar een array
    return x


def powerdp(array_frequenties):
    if (venster):
        array_frequenties = array_frequenties * np.hanning(AANTAL_FREQUENTIES)  # windowing van frequentiekarakteristiek
    inverse = np.fft.ifft(array_frequenties)  # neem inverse ft van frequentiekarakteristiek
    pdp = np.sqrt((np.abs(inverse) ** 2) / AANTAL_FREQUENTIES)  # freq_kar omzetten in PDP (RMS)
    return pdp


def channel2APDP(PUNT, plotten=False):
    freq_kar = np.zeros(AANTAL_FREQUENTIES, dtype=complex)  # lege array voor frequentiekarakteristiek te bewaren
    apdp = np.zeros(AANTAL_FREQUENTIES, dtype=complex)  # lege array voor de PDP's te bewaren

    # voor elke meting de PDP berekenen en bewaren in de array APDP
    for meting in range(AANTAL_METINGEN):
        for freq in range(AANTAL_FREQUENTIES):
            freq_kar[freq] = data[freq][PUNT][meting]
        pdp = 20 * np.log10(powerdp(freq_kar))
        for i in range(AANTAL_FREQUENTIES):
            apdp[i] += pdp[i]
    apdp /= AANTAL_METINGEN  # de gemiddelde van de pdp's berekenen
    if plotten:
        plt.plot(apdp)
        plt.show()
    return apdp


def APDP2delays(APDP):
    APDP[APDP < offset] = offset  # ruis in vlakke stukken elimineren
    T = sig.argrelmax(APDP)  # indexen van de pieken bepalen
    return T[0][0] * stap, T[0][1] * stap  # index * tijd tussen 2 frequenties om tau te bepalen


# voor elk punt tau0 en tau1 bepalen
def delays_berekenen():
    delays = np.zeros((AANTAL_PUNTEN, 2))
    for PUNT in range(AANTAL_PUNTEN):
        APDP = channel2APDP(PUNT)
        delays[PUNT] = APDP2delays(APDP)
    return delays

def calculate_location(tau0, tau1):
    a = tau1 * cst.speed_of_light
    b = tau0 * cst.speed_of_light
    y = (a ** 2 - b ** 2) / 4
    x = np.sqrt(-(a) ** 4 + 2 * a ** 2 * b ** 2 + 8 * a ** 2 - b ** 4 + 8 * b ** 2 - 16) / 4
    return x, y


def fout_berekenen():
    fout_x = []
    fout_y = []
    fout = []
    for i in range(AANTAL_PUNTEN):  # fout op x en y respectievelijk bepalen van elk punt
        fout_x.append(x_coordinaat[i] - x_juist[i])
        fout_y.append(y_coordinaat[i] - y_juist[i])

    for i in range(AANTAL_PUNTEN):  # algemene fout bepalen van elk punt
        fout.append(np.sqrt(fout_x[i] ** 2 + fout_y[i] ** 2))

    list.sort(fout)
    mediaan = (fout[11] + fout[12]) / 2
    print("fout = ", mediaan)

# de juiste x waarden bewaren
def x_waarden():
    x_array = []
    for i in range(0, AANTAL_PUNTEN):
        theta = i * 15
        xn = 4 + 3 * math.sin(theta)
        x_array.append(xn)
    return x_array

# de juiste y waarden bewaren
def y_waarden():
    y_array = []
    for i in range(0, AANTAL_PUNTEN):
        theta = i * 15
        yn = 4 + 3 * math.sin(theta) * math.cos(theta)
        y_array.append(yn)
    return y_array


def coordinaten_echte_traject():
    x_voorlopig = x_waarden()  # Omdat via de methode x_waarden() en y_waarden() de coordinaten van het
    y_voorlopig = y_waarden()  # juiste pad niet overeenkomen met de volgorde van de punten van het gerealiseerde
    volgorde = [0, 13, 8, 21, 3, 16, 11, 6, 19, 1, 14, 9, 22, 4, 17, 12, 7, 20, 2, 15, 10, 23, 5,
                18]  # pad, hebben we hier manueel de punten van het juiste pad in de juiste volgorde gezet
    y_juist = []
    x_juist = []
    for i in volgorde:
        x_juist.append(x_voorlopig[i])
        y_juist.append(y_voorlopig[i])
    return x_juist, y_juist


# Main
venster = 1  # Venster aan- of uitzetten: 1 = aan / 0 = uit
dataset = 1

for dataset in range(2):    # het traject plotten voor elk dataset
    if (dataset == 1):
        print("\nDataset 1: ")
        data = data_verwerken('Dataset_1.mat')
        stap = 1 / 10 / (10 ** 6) / 200
        offset = -100
        titel = "Dataset 1"     # titel van de plot
    else:
        print("Dataset 2: ")
        data = data_verwerken('Dataset_2.mat')
        stap = 1 / 10 / (10 ** 6) / 1000
        offset = -120
        titel = "Dataset 2"

    AANTAL_FREQUENTIES = len(data)
    AANTAL_PUNTEN = len(data[0])
    AANTAL_METINGEN = len(data[0][0])

    tau_waarden = delays_berekenen()    # array met tau0 en tau1 van elk punt

    x_coordinaat = []
    y_coordinaat = []

    for i in range(0, AANTAL_PUNTEN):   # de locatie van elk punt bepalen
        coord = calculate_location(tau_waarden[i][0], tau_waarden[i][1])
        print('Punt', i, ':', coord)
        x_coordinaat.append(coord[0])
        y_coordinaat.append(coord[1])

    x_juist, y_juist = coordinaten_echte_traject()

    fout_berekenen()

    # het traject plotten
    plt.scatter(x_coordinaat, y_coordinaat, color='darkorange')
    plt.plot(x_coordinaat, y_coordinaat, color='darkorange')
    plt.scatter(x_juist, y_juist, color='blue')
    plt.plot(x_juist, y_juist, color='blue')
    plt.title(titel)

    orange_patch = mpatches.Patch(color='darkorange', label='Gereconstrueerd traject')
    blue_patch = mpatches.Patch(color='blue', label='Echte traject')
    plt.legend(handles=[blue_patch, orange_patch])
    plt.show()
