import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

c = 300 * 10**6  # m/s
freq = 24 * 10**9  # Hz
wavelength = c / freq

aCount = 18
phaseOffset = np.pi * .3
posOffset = wavelength
antennas = [(x - (aCount - 1) * posOffset / 2, 0)
            for x in np.arange(0, aCount * posOffset, posOffset)]
aPhases = phaseOffset * np.arange(aCount)


def power(phases):
    return sum(np.cos(phases))**2 + sum(np.sin(phases))**2


def phases(distances):
    return aPhases + 2 * np.pi / wavelength * distances


def distances(pos):
    return np.array([distance.euclidean(a, pos) for a in antennas])


samples = np.linspace(-aCount * posOffset / 1.5, aCount * posOffset / 1.5, 100)
grid = [(x, y) for x in samples for y in samples]
x = [p[0] for p in grid]
y = [p[1] for p in grid]

plt.figure()
plt.scatter([a[0] for a in antennas], [a[1] for a in antennas], marker="x")
plt.scatter(x, y, c=[power(phases(distances(p))) for p in grid])
plt.show()
