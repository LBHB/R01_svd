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

val = 'dp_diag'

mask = ~df.tdr_overall & ~df.pca & df.tar_tar & (df.area=='A1')
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

plt.show()