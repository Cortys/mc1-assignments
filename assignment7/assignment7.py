import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generateData(size):
    return np.random.randint(2, size=size)


def QPSK(data):
    return [np.complex(data[i] * 2 - 1, (data[i + 1] * 2 - 1))
            for i in range(0, len(data), 2)]


def deQPSK(symbols):
    return reduce((lambda bits, symbol: bits + [np.real(symbol) > 0, np.imag(symbol) > 0]),
                  symbols, [])


def generateChannels(nt, nr, meanAttenuation):
    return np.multiply(np.random.exponential(scale=meanAttenuation, size=(nr, nt)),
                       np.exp(np.complex(0, 1) * 2 * np.pi * np.random.random((nr, nt))))


def transmit(symbols, channels, noise):
    return np.dot(channels, symbols) \
        + np.random.normal(scale=noise, size=np.shape(channels)[0])


def simulateTransmission(channelCount, snr):
    channels = generateChannels(channelCount, channelCount, 1)
    U, s, V = np.linalg.svd(channels)
    S = np.diag(1 / s)
    UH = np.array(np.matrix(U).H)
    VH = np.array(np.matrix(V).H)
    data = generateData(2 * channelCount)
    modulatedData = QPSK(data)
    sendSymbols = np.dot(VH, modulatedData)
    transmission = transmit(sendSymbols, channels, 1 / snr)
    receivedSymbols = np.dot(S, np.dot(UH, transmission))
    receivedData = deQPSK(receivedSymbols)
    return np.sum(np.abs(data - receivedData)) / len(data)


def simulateTransmissions(n, channelCount, snr):
    return np.sum([simulateTransmission(channelCount, snr) for i in range(n)]) / n


channelCounts = np.arange(1, 11)
snrs = np.linspace(-10, 1)

x, y = np.meshgrid(channelCounts, snrs)
data = np.array([simulateTransmissions(10, channelCount, 10 ** snr) for channelCount, snr in zip(np.ravel(x), np.ravel(y))])
z = data.reshape(x.shape)

fig = plt.figure()
ax = Axes3D(fig)

ax.plot_surface(x, y, z)
plt.show()
