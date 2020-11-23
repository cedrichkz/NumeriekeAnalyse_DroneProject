import scipy.io as sc
import scipy.fft
import numpy
import matplotlib.pyplot as plt
from math import sqrt, sin, cos

def data_1_verwerken(file):
    mat = sc.loadmat(file)

    new = numpy.zeros((200, 24), dtype=complex)
    x = numpy.array(mat.get('H'))


    som=0
    for k in range(200):
        for i in range(24):
            for j in range(100):
                som += x[k, i, j]
            new[k, i] = som/100
    return new



def channel2APDP(ar):
    ar2 = scipy.fft.ifft(ar)
    for i in range(len(ar)):
        ar2[i] = ar[i]*ar[i]
    return ar2

def APDP2delays(APDP):
    T0 = numpy.argmax(APDP)
    APDP[T0] = 0
    T1 = numpy.argmax(APDP)
    T = {T0, T1}
    return T


ar = data_1_verwerken('Dataset_1.mat')
delays = []
for i in range(24):
    APDP = channel2APDP(ar[:,i])
    delays.append(APDP2delays(APDP))


#functie om posities te bepalen
def calculate_location(t0, t1):
    v_licht = None  #lichtsnelheid
    hoek_n = 0      #beginpositie
    locaties = []   #lijst waarin de posities komen

    #loop waarmee de 24 posities worden berekend
    for x in range(24):

        #berekening x- en y coordinaat
        x_n = sqrt( (v_licht*t0)**2 - (4+3*sin(hoek_n)*sin(hoek_n) - 1)**2  )
        y_n = sqrt( (v_licht*t0)**2 - (4+3*sin(hoek_n))**2 ) + 1
       
        pos_n = (x_n, y_n)          
        locaties.append(pos_n)      #coordinaten toevoegen aan lijst
        hoek_n += 15                #volgende vliegtuigpositie
    
