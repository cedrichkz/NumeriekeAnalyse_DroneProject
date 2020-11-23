import scipy.io as sc
import scipy.fft
import numpy
import matplotlib.pyplot as plt

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
    T0Y = max(APDP)
    T0X = APDP.index(T0Y)
    T0 = {T0X, T0Y}
    hulp = []
    for i in range(T0X+1, len(APDP)):
        hulp[i-T0X-1] = APDP[i]
    T1Y = max(hulp)
    T1X = hulp.index(T1Y)
    T1 = {T1X, T1Y}
    return {T0, T1}

ar = data_1_verwerken('Dataset_1.mat')
delays = []
for i in range(24):
    APDP = channel2APDP(ar[:,i])
    delays.append(APDP2delays(APDP))
