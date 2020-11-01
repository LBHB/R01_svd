"""
Compare first-order simulation to raw data
"""
from settings import DIR
import nems.db as nd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['pdf.fonttype'] = 42

df = pd.read_pickle(DIR + 'results/res.pickle')
df.index = df.pair

val = 'dp_opt'

mask = ~df.tdr_overall & ~df.pca & df.cat_tar & (df.batch.isin([324 & 325]))
mask_raw = mask & ~df.sim1
mask_sim = mask & df.sim1

f, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

# raw data, active vs. passive
ax[0].scatter(df[mask_raw & ~df.active][val], df[mask_raw & df.active][val], edgecolor='white')
ax[0].set_xlabel('Passive')
ax[0].set_ylabel('Active')
ax[0].set_title('Raw Data')

# simulated (first order only) data, active vs. passive
ax[1].scatter(df[mask_sim & ~df.active][val], df[mask_sim & df.active][val], edgecolor='white')
ax[1].set_xlabel('Passive')
ax[1].set_ylabel('Active')
ax[1].set_title('First-order simulation')

mi = np.min(ax[0].get_xlim()+ax[0].get_ylim())
ma = np.max(ax[0].get_xlim()+ax[0].get_ylim())
ax[0].plot([mi, ma], [mi, ma], 'k--')
ax[1].plot([mi, ma], [mi, ma], 'k--')

f.tight_layout()


# compare difference directly for active / passive
f, ax = plt.subplots(1, 1, figsize=(4, 4))

ax.scatter(df[mask_raw & df.active][val] - df[mask_raw & ~df.active][val], 
                   df[mask_sim & df.active][val] - df[mask_sim & ~df.active][val], edgecolor='white')
ax.set_xlabel('Raw Data')
ax.set_ylabel('First-order simulation')
ax.set_title(r"$\Delta d'^2$")
mi = np.min(ax.get_xlim()+ax.get_ylim())
ma = np.max(ax.get_xlim()+ax.get_ylim())
ax.plot([mi, ma], [mi, ma], 'k--')

f.tight_layout()

# if ref vs tar comparison have two dims to look at:
    # diff between CFs of sounds, and SNR of target
    # for these two dims, look at "redisual" change in decoding (presumably due to 2nd order)
norm = True
if df[mask_raw].ref_tar.sum() == df[mask_raw].shape[0]:
    f, ax = plt.subplots(1, 1, figsize=(6, 6))
    if norm:
        residual = pd.DataFrame(((df[mask_raw & df.active][val] - df[mask_raw & ~df.active][val]) / (df[mask_raw & df.active][val] + df[mask_raw & ~df.active][val])) - \
                        ((df[mask_sim & df.active][val] - df[mask_sim & ~df.active][val]) / (df[mask_sim & df.active][val] + df[mask_sim & ~df.active][val])))
    else:
        residual = pd.DataFrame((df[mask_raw & df.active][val] - df[mask_raw & ~df.active][val]) - \
                        (df[mask_sim & df.active][val] - df[mask_sim & ~df.active][val]))        
    residual.at[:, 'sep'] = abs(np.log2(df[mask_raw & df.active]['f1'] / df[mask_raw & df.active]['f2']))
    residual['sep'] = residual['sep'].apply(lambda x: x - x % 0.5)
    residual.at[:, 'tar_snr'] = df[mask_raw & df.active]['snr2']

    for tsnr in residual.tar_snr.unique():
        dat = residual[residual.tar_snr==tsnr].groupby(by='sep').mean()
        dsem = residual[residual.tar_snr==tsnr].groupby(by='sep').sem()
        idx = np.argwhere(residual[residual.tar_snr==tsnr].groupby(by='sep').count()[val].values >= 5).squeeze()

        #if tsnr != np.inf:
        ax.errorbar(dat.iloc[idx].index, dat.iloc[idx][val].values, yerr=dsem.iloc[idx][val], label=tsnr, lw=2)

    ax.set_title('Reference vs. Target')
    ax.legend(frameon=False)
    ax.set_xlabel('Frequency separation (octaves)')
    ax.set_ylabel(r"Residual $\Delta d'$"+'\n'+ r"(Raw $\Delta d'$ - First-order $\Delta d'$")
    ax.axhline(0, linestyle='--', color='grey', lw=2)
    f.tight_layout()

plt.show()