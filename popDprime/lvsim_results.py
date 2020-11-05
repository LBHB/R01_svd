"""
Compare decoding results for latent variable simulations to the results for 
the raw data.
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
mpl.rcParams['font.size'] = 6

val = 'dp_opt'
ms = 30
ma = 40

df = pd.read_pickle(DIR + 'results/res.pickle')
df.index = df.pair
df = df[df.batch.isin([324, 325])]
df[val] = np.sqrt(df[val])

df_psth = pd.read_pickle(DIR + 'results/res_predOnly.pickle')
df_psth.index = df_psth.pair
df_psth[val] = np.sqrt(df_psth[val])

df_ind = pd.read_pickle(DIR + 'results/res_indNoise.pickle')
df_ind.index = df_psth.pair
df_ind[val] = np.sqrt(df_ind[val])

df_lv = pd.read_pickle(DIR + 'results/res_indNoiseLV.pickle')
df_lv.index = df_lv.pair
df_lv[val] = np.sqrt(df_lv[val])

tar_mask = (df.tar_tar) & (df.tdr_overall) & (~df.pca) & df.batch.isin([324, 325]) & (df.f1 == df.f2) & ~df.sim1
cat_mask = (df.cat_tar) & (df.tdr_overall) & (~df.pca) & df.batch.isin([324, 325]) & (df.f1 == df.f2) & ~df.sim1
ref_mask = (df.ref_ref) & (df.tdr_overall) & (~df.pca) & df.batch.isin([324, 325]) & ~df.sim1

# first, scatter plot of active/passive results for all models (and raw data)
f, ax = plt.subplots(1, 4, figsize=(8, 2))

for a, d, lab in zip(ax.flatten(), [df_psth, df_ind, df_lv, df], ['pred', 'indNoise', 'LV', 'Raw']):

    a.scatter(d[tar_mask & ~df.active].groupby(by='site').mean()[val], 
                d[tar_mask & df.active].groupby(by='site').mean()[val], s=ms, edgecolor='k', color='coral', label='Tar vs. Tar')
    a.scatter(d[cat_mask & ~df.active].groupby(by='site').mean()[val], 
            d[cat_mask & df.active].groupby(by='site').mean()[val], s=ms, edgecolor='k', color='lightgrey', label='Cat vs. Tar')
    a.scatter(d[ref_mask & ~df.active].groupby(by='site').mean()[val], 
            d[ref_mask & df.active].groupby(by='site').mean()[val], s=ms, edgecolor='k', color='mediumblue', label='Ref vs. Ref')
    a.set_xlabel(r"$d'$ Active")
    a.set_ylabel(r"$d'$ Passive")
    ma = np.max(a.get_ylim()+a.get_xlim())
    a.plot([0, ma], [0, ma], '--', color='grey', lw=2)
    a.set_title(lab)

a.legend(frameon=False)

f.tight_layout()


# concise summary of results for each model (and raw data) -- might fit better on the grant

plt.show()