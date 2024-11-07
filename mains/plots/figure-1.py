import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = '/N/slate/cwseitz/cvdm/Sim/4x/Sim-2/eval_data/'
df100 = pd.read_csv(path + 'N100-error.csv')
df200 = pd.read_csv(path + 'N200-error.csv')
df500 = pd.read_csv(path + 'N500-error.csv')

def compute_stats(df):
    grouped = df.groupby(['label', 'prefix', 'idx'])
    xacc = [np.mean(group['x_err'].values) for _, group in grouped]
    yacc = [np.mean(group['y_err'].values) for _, group in grouped]
    N0 = [group['N0'].values[0] for _, group in grouped]
    return np.array(N0), np.array(xacc), np.array(yacc)

N0_100, xacc_100, yacc_100 = compute_stats(df100)
N0_200, xacc_200, yacc_200 = compute_stats(df200)
N0_500, xacc_500, yacc_500 = compute_stats(df500)

pixel_size = 25.0
bins = np.arange(100, 1000, 100)

def bin_std(N0, acc):
    return np.array([np.std(acc[(N0 >= bins[i]) & (N0 < bins[i+1])]) for i in range(len(bins) - 1)])

xstd_binned_100 = bin_std(N0_100, xacc_100)
ystd_binned_100 = bin_std(N0_100, yacc_100)
xstd_binned_200 = bin_std(N0_200, xacc_200)
ystd_binned_200 = bin_std(N0_200, yacc_200)
xstd_binned_500 = bin_std(N0_500, xacc_500)
ystd_binned_500 = bin_std(N0_500, yacc_500)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

# Plot for x-accuracy standard deviations
ax1.plot(bins[:-1], xstd_binned_100 * pixel_size, 'x', color='red', label=r'$\rho = 100$')
ax1.plot(bins[:-1], xstd_binned_100 * pixel_size, '--', color='red', alpha=0.3)
ax1.plot(bins[:-1], xstd_binned_200 * pixel_size, 'x', color='blue', label=r'$\rho = 200$')
ax1.plot(bins[:-1], xstd_binned_200 * pixel_size, '--', color='blue', alpha=0.3)
ax1.plot(bins[:-1], xstd_binned_500 * pixel_size, 'x', color='pink', label=r'$\rho = 500$')
ax1.plot(bins[:-1], xstd_binned_500 * pixel_size, '--', color='pink', alpha=0.3)
ax1.set_xscale('log')
ax1.set_xticks([100, 500, 1000])
ax1.set_xticklabels([r'$10^2$', r'$5 \times 10^2$', r'$10^3$'])
ax1.set_xlabel('Photons')
ax1.set_ylabel('Localization error (nm)')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(frameon=False)

# Plot for y-accuracy standard deviations
ax2.plot(bins[:-1], ystd_binned_100 * pixel_size, 'x', color='red', label=r'$\rho = 100$')
ax2.plot(bins[:-1], ystd_binned_100 * pixel_size, '--', color='red', alpha=0.3)
ax2.plot(bins[:-1], ystd_binned_200 * pixel_size, 'x', color='blue', label=r'$\rho = 200$')
ax2.plot(bins[:-1], ystd_binned_200 * pixel_size, '--', color='blue', alpha=0.3)
ax2.plot(bins[:-1], ystd_binned_500 * pixel_size, 'x', color='pink', label=r'$\rho = 500$')
ax2.plot(bins[:-1], ystd_binned_500 * pixel_size, '--', color='pink', alpha=0.3)
ax2.set_xscale('log')
ax2.set_xticks([100, 500, 1000])
ax2.set_xticklabels([r'$10^2$', r'$5 \times 10^2$', r'$10^3$'])
ax2.set_xlabel('Photons')
ax2.set_ylabel('Localization error (nm)')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(frameon=False)

plt.tight_layout()
plt.show()

