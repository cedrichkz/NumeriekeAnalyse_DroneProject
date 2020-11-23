from math import *
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.fftpack
import math

METINGEN=100
PUNTEN=24

Dataset1=sio.loadmat('Dataset_1.mat')['H']
Dataset2=sio.loadmat('Dataset_2.mat')['H']



def calculate_delays(frequenties,Dataset,Venster):
    Apdp_per_positie=[]
    Tau0=[]
    Tau1=[]
    for i in range(0,PUNTEN):
        Apdp_per_positie.append(channel2APDP(i,frequenties,Dataset,Venster))
        x=APDP2delays(Apdp_per_positie[i],frequenties)
        Tau0.append(x[0])
        Tau1.append(x[1])
    Tau=Tau0,Tau1
    return Tau;


def channel2APDP(positie,frequenties,Dataset,Venster):  #APDP berekenen
    pdp=[] #Array met 100 pdp's per element voor elke frequentie
    frequentiekarakteristiek=np.zeros(frequenties,dtype=np.complex) #Verzameling van alle frequenties voor 1 bepaalde meting
    apdp=[] #Array met het gemiddelde van de pdp's voor elke frequentie 
    for meting in range(0,METINGEN): 
        for frequentie in range(0,frequenties):    
            frequentiekarakteristiek[frequentie]=Dataset[frequentie][positie][meting]
        if(Venster==1):
            frequentiekarakteristiek=frequentiekarakteristiek*signal.gaussian(frequenties,26) #Window
        pdp.append(np.sqrt(abs(np.fft.ifft((frequentiekarakteristiek)))))
    apdp =20*np.log10(np.mean(pdp,axis=0))
    return apdp;

def APDP2delays(apdp,frequenties):   #T0 en T1 bepalen 
    tau=[]
    ind_piek1=apdp.tolist().index(max(apdp)) #index van eerste piek (T0)
    piek2=-100
    for i in range(ind_piek1+1,frequenties): #Vanaf T0 bekijken we wanneer er terug een stijging is
        if apdp[i]>apdp[i-1] and apdp[i]>piek2:
            piek2=apdp[i]
    tau=ind_piek1,apdp.tolist().index(piek2) #index van tweede piek (T1)
    return tau;

def x_exact():
    x = []
    for i in range(0,PUNTEN):
        theta = i*15
        xn_exact = 3 + theta/90
        x.append(xn_exact)
    return x;

def y_exact():
    y = []
    for i in range(0,PUNTEN):
        theta = i*15
        yn_exact = 3+(2-3/48*i)*math.sin(np.deg2rad(theta))
        y.append(yn_exact)
    return y;


def calculate_constant(x0,y0,tau):
    snelheid = math.sqrt((x0**2)+((y0-1)**2))/tau
    return snelheid;

def calculate_location(tau0,tau1,snelheid):
    y = (((tau1**2)-(tau0**2))*(snelheid**2))/4
    x = math.sqrt((tau0*snelheid)**2-(y-1)**2)
    coord = x,y
    return coord;

def gemiddelde_fout(x_exact,y_exact,x_berekend,y_berekend):
    fout_x=[]
    fout_y=[]
    for i in range(0,PUNTEN):
        fout_x.append(x_berekend[i]-x_exact[i])
        fout_y.append(y_berekend[i]-y_exact[i])
    #fout=fout_x,fout_y
    #for i in range(0,PUNTEN):
        #print('[',fout[0][i],fout[1][i],']')


#KIES HIER DE OPTIES
Dataset=1   #   1: dataset1         2: dataset2
Venster=0   #   0: zonder venster   1: met venster

#main methode
x=x_exact()
y=y_exact()
if(Dataset==1):
    Tau = calculate_delays(201,Dataset1,Venster)
    snelheid = calculate_constant(x[0],y[0],Tau[0][0])
    x_berekend = []
    y_berekend = []
    for i in range(0,PUNTEN):
        coord = calculate_location(Tau[0][i],Tau[1][i],snelheid)
        print('Punt',i,':',coord)
        x_berekend.append(coord[0])
        y_berekend.append(coord[1])
    print("Fout:")
    gemiddelde_fout(x,y,x_berekend,y_berekend)
    plt.plot(x_berekend,y_berekend)
else:
    Tau = calculate_delays(1001,Dataset2,Venster)
    snelheid = calculate_constant(x[0],y[0],Tau[0][0])
    x_berekend = []
    y_berekend = []
    for i in range(0,PUNTEN):
        coord = calculate_location(Tau[0][i],Tau[1][i],snelheid)
        print('Punt',i,':',coord)
        x_berekend.append(coord[0])
        y_berekend.append(coord[1])
    print("Fout:")
    gemiddelde_fout(x,y,x_berekend,y_berekend)
    plt.plot(x_berekend,y_berekend)
plt.plot(x,y)
plt.show()





