"""
First-order vs. second order contributions to d' changes
First-order delta dprime vs. raw delta dprime. For PEG / A1
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

savefig = True

figsave = DIR + 'results/figures/first_second_order.pdf'

df = pd.read_pickle(DIR + 'results/res.pickle')
df.index = df.pair

cmap = {
    'cat_tar': 'lightgrey',
    'tar_tar': 'coral',
    'ref_ref': 'mediumblue'
}

val = 'dp_opt'
df[val] = np.sqrt(df[val])
m = 4
mi = -1
ms = 30

tar_mask = (df.tar_tar) & (df.tdr_overall==True) & (~df.pca) & df.batch.isin([324, 325]) & (df.f1 == df.f2) 
cat_mask = (df.cat_tar) & (df.tdr_overall==True) & (~df.pca) & df.batch.isin([324, 325]) & (df.f1 == df.f2) 
ref_mask = (df.ref_ref) & (df.tdr_overall==True) & (~df.pca) & df.batch.isin([324, 325])

# compute delta dprime for simulations and for raw data, for each category, for each area
delta = dict.fromkeys(['A1', 'PEG'])
sem = dict.fromkeys(['A1', 'PEG'])
delta_sim = dict.fromkeys(['A1', 'PEG'])
sem_sim = dict.fromkeys(['A1', 'PEG'])
for area in ['A1', 'PEG']:
    delta[area] = {}
    sem[area] = {}
    delta_sim[area] = {}
    sem_sim[area] = {}
    for category, mask in zip(['tt', 'ct', 'rr'], [tar_mask, cat_mask, ref_mask]):
        # raw data
        act = df[mask & df.active & (df.area==area) & ~df.sim1][[val, 'site']].set_index('site')
        pas = df[mask & ~df.active & (df.area==area) & ~df.sim1][[val, 'site']].set_index('site')
        if norm:
            n = (act + pas)
        else:
            n = 1
        delta[area][category] = ((act - pas) / n).groupby(level=0).mean().values
        sem[area][category] =  ((act - pas) / n).groupby(level=0).sem().values

        # sim1 data
        act = df[mask & df.active & (df.area==area) & df.sim1][[val, 'site']].set_index('site')
        pas = df[mask & ~df.active & (df.area==area) & df.sim1][[val, 'site']].set_index('site')
        if norm:
            n = (act + pas)
        else:
            n = 1
        delta_sim[area][category] = ((act - pas) / n).groupby(level=0).mean().values
        sem_sim[area][category] =  ((act - pas) / n).groupby(level=0).sem().values

f, ax = plt.subplots(2, 1, figsize=(2, 4))

# ==== A1 ===== 
ax[0].scatter(delta_sim['A1']['tt'], delta['A1']['tt'], s=ms, color=cmap['tar_tar'], edgecolor='k', label='Tar vs. Tar')
ax[0].scatter(delta_sim['A1']['ct'], delta['A1']['ct'], s=ms, color=cmap['cat_tar'], edgecolor='k', label='Tar vs. Tar')
ax[0].scatter(delta_sim['A1']['rr'], delta['A1']['rr'], s=ms, color=cmap['ref_ref'], edgecolor='k', label='Ref vs. Ref')
ax[0].set_xlabel(r"$\Delta d'$ 1st-order")
ax[0].set_ylabel(r"$\Delta d'$ Raw")
ax[0].plot([mi, m], [mi, m], '--', color='grey', lw=2)
ax[0].set_title('A1')
ax[0].legend(frameon=False)

# ==== PEG ===== 
ax[1].scatter(delta_sim['PEG']['tt'], delta['PEG']['tt'], s=ms, color=cmap['tar_tar'], edgecolor='k', label='Tar vs. Tar')
ax[1].scatter(delta_sim['PEG']['ct'], delta['PEG']['ct'], s=ms, color=cmap['cat_tar'], edgecolor='k', label='Tar vs. Tar')
ax[1].scatter(delta_sim['PEG']['rr'], delta['PEG']['rr'], s=ms, color=cmap['ref_ref'], edgecolor='k', label='Ref vs. Ref')
ax[1].set_xlabel(r"$\Delta d'$ 1st-order")
ax[1].set_ylabel(r"$\Delta d'$ Raw")
ax[1].plot([mi, m], [mi, m], '--', color='grey', lw=2)
ax[1].set_title('PEG')

f.tight_layout()

plt.show()