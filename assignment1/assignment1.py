import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Data for plotting
x = np.linspace(-5, 5, 100)
samples1 = stats.norm.rvs(0, 1, 50000)
samples2 = stats.norm.rvs(0, 1, 50000)
normal = stats.norm.pdf(x)

fig, axs = plt.subplots(2, 1)

axs[0].hist(samples1, bins=np.arange(-5, 5, .1), normed=1)
axs[0].plot(x, normal)

axs[1].hist2d(samples1, samples2, bins=np.arange(-5, 5, .1), normed=1)

plt.show()
