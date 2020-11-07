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

savefig = True

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
df_ind.index = df_ind.pair
df_ind[val] = np.sqrt(df_ind[val])

df_lv = pd.read_pickle(DIR + 'results/res_indNoiseLV.pickle')
df_lv.index = df_lv.pair
df_lv[val] = np.sqrt(df_lv[val])

tar_mask = (df.tar_tar) & (df.tdr_overall) & (~df.pca) & df.batch.isin([324, 325]) & (df.f1 == df.f2) & ~df.sim1
cat_mask = (df.cat_tar) & (df.tdr_overall) & (~df.pca) & df.batch.isin([324, 325]) & (df.f1 == df.f2) & ~df.sim1
ref_mask = (df.ref_ref) & (df.tdr_overall) & (~df.pca) & df.batch.isin([324, 325]) & ~df.sim1

# first, scatter plot of active/passive results for all models (and raw data)
f, ax = plt.subplots(1, 4, figsize=(8, 2))
ms = 30

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
xticks = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
xtickm = [1, 5, 9, 13]
ms = 5

f, ax = plt.subplots(1, 1, figsize=(2, 2))

# clean prediction
ax.errorbar(xticks[0], df_psth[ref_mask & ~df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df_psth[ref_mask & ~df.active].groupby(by='site').mean()[val].sem(), 
                               markerfacecolor='none', color='mediumblue', ecolor='mediumblue', capsize=3, marker='o', markersize=ms)
ax.errorbar(xticks[0], df_psth[ref_mask & df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df_psth[ref_mask & df.active].groupby(by='site').mean()[val].sem(), 
                               color='mediumblue', ecolor='mediumblue', capsize=3, marker='o', markersize=ms)
ax.errorbar(xticks[1], df_psth[cat_mask & ~df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df_psth[cat_mask & ~df.active].groupby(by='site').mean()[val].sem(), 
                               markerfacecolor='none', color='lightgrey', ecolor='lightgrey', capsize=3, marker='o', markersize=ms)
ax.errorbar(xticks[1], df_psth[cat_mask & df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df_psth[cat_mask & df.active].groupby(by='site').mean()[val].sem(), 
                               color='lightgrey', ecolor='lightgrey', capsize=3, marker='o', markersize=ms)
ax.errorbar(xticks[2], df_psth[tar_mask & ~df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df_psth[tar_mask & ~df.active].groupby(by='site').mean()[val].sem(), 
                               markerfacecolor='none', color='coral', ecolor='coral', capsize=3, marker='o', markersize=ms)
ax.errorbar(xticks[2], df_psth[tar_mask & df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df_psth[tar_mask & df.active].groupby(by='site').mean()[val].sem(), 
                               color='coral', ecolor='coral', capsize=3, marker='o', markersize=ms)
# independent noise
ax.errorbar(xticks[3], df_ind[ref_mask & ~df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df_ind[ref_mask & ~df.active].groupby(by='site').mean()[val].sem(), 
                               markerfacecolor='none', color='mediumblue', ecolor='mediumblue', capsize=3, marker='o', markersize=ms)
ax.errorbar(xticks[3], df_ind[ref_mask & df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df_ind[ref_mask & df.active].groupby(by='site').mean()[val].sem(), 
                               color='mediumblue', ecolor='mediumblue', capsize=3, marker='o', markersize=ms)
ax.errorbar(xticks[4], df_ind[cat_mask & ~df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df_ind[cat_mask & ~df.active].groupby(by='site').mean()[val].sem(), 
                               markerfacecolor='none', color='lightgrey', ecolor='lightgrey', capsize=3, marker='o', markersize=ms)
ax.errorbar(xticks[4], df_ind[cat_mask & df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df_ind[cat_mask & df.active].groupby(by='site').mean()[val].sem(), 
                               color='lightgrey', ecolor='lightgrey', capsize=3, marker='o', markersize=ms)
ax.errorbar(xticks[5], df_ind[tar_mask & ~df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df_ind[tar_mask & ~df.active].groupby(by='site').mean()[val].sem(), 
                               markerfacecolor='none', color='coral', ecolor='coral', capsize=3, marker='o', markersize=ms)
ax.errorbar(xticks[5], df_ind[tar_mask & df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df_ind[tar_mask & df.active].groupby(by='site').mean()[val].sem(), 
                               color='coral', ecolor='coral', capsize=3, marker='o', markersize=ms)

# independent noise + LV
ax.errorbar(xticks[6], df_lv[ref_mask & ~df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df_lv[ref_mask & ~df.active].groupby(by='site').mean()[val].sem(), 
                               markerfacecolor='none', color='mediumblue', ecolor='mediumblue', capsize=3, marker='o', markersize=ms)
ax.errorbar(xticks[6], df_lv[ref_mask & df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df_lv[ref_mask & df.active].groupby(by='site').mean()[val].sem(), 
                               color='mediumblue', ecolor='mediumblue', capsize=3, marker='o', markersize=ms)
ax.errorbar(xticks[7], df_lv[cat_mask & ~df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df_lv[cat_mask & ~df.active].groupby(by='site').mean()[val].sem(), 
                               markerfacecolor='none', color='lightgrey', ecolor='lightgrey', capsize=3, marker='o', markersize=ms)
ax.errorbar(xticks[7], df_lv[cat_mask & df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df_lv[cat_mask & df.active].groupby(by='site').mean()[val].sem(), 
                               color='lightgrey', ecolor='lightgrey', capsize=3, marker='o', markersize=ms)
ax.errorbar(xticks[8], df_lv[tar_mask & ~df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df_lv[tar_mask & ~df.active].groupby(by='site').mean()[val].sem(), 
                               markerfacecolor='none', color='coral', ecolor='coral', capsize=3, marker='o', markersize=ms)
ax.errorbar(xticks[8], df_lv[tar_mask & df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df_lv[tar_mask & df.active].groupby(by='site').mean()[val].sem(), 
                               color='coral', ecolor='coral', capsize=3, marker='o', markersize=ms)

# Actual data
ax.errorbar(xticks[9], df[ref_mask & ~df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df[ref_mask & ~df.active].groupby(by='site').mean()[val].sem(), 
                               markerfacecolor='none', color='mediumblue', ecolor='mediumblue', capsize=3, marker='o', markersize=ms)
ax.errorbar(xticks[9], df[ref_mask & df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df[ref_mask & df.active].groupby(by='site').mean()[val].sem(), 
                               color='mediumblue', ecolor='mediumblue', capsize=3, marker='o', markersize=ms)
ax.errorbar(xticks[10], df[cat_mask & ~df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df[cat_mask & ~df.active].groupby(by='site').mean()[val].sem(), 
                               markerfacecolor='none', color='lightgrey', ecolor='lightgrey', capsize=3, marker='o', markersize=ms)
ax.errorbar(xticks[10], df[cat_mask & df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df[cat_mask & df.active].groupby(by='site').mean()[val].sem(), 
                               color='lightgrey', ecolor='lightgrey', capsize=3, marker='o', markersize=ms)
ax.errorbar(xticks[11], df[tar_mask & ~df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df[tar_mask & ~df.active].groupby(by='site').mean()[val].sem(), 
                               markerfacecolor='none', color='coral', ecolor='coral', capsize=3, marker='o', markersize=ms)
ax.errorbar(xticks[11], df[tar_mask & df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df[tar_mask & df.active].groupby(by='site').mean()[val].sem(), 
                               color='coral', ecolor='coral', capsize=3, marker='o', markersize=ms)

ax.set_xlim((xticks[0]-1, xticks[-1]+1))
ax.set_xticks(xtickm)
ax.set_xticklabels(['Clean', 'Indep', 'Indep + LV', 'Actual'], rotation=45)
ax.set_xlabel('Dataset')
ax.set_ylabel(r"$d'$")

f.tight_layout()


# group sound categories for single dprime measurement
ms = 5

f, ax = plt.subplots(1, 1, figsize=(2, 2))

# clean pred
ax.errorbar(0, df_psth[(tar_mask | ref_mask | cat_mask) & ~df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df[tar_mask & ~df.active].groupby(by='site').mean()[val].sem(), 
                               color='grey', ecolor='grey', capsize=3, marker='o', markersize=ms, label='Passive')
ax.errorbar(0, df_psth[(tar_mask | ref_mask | cat_mask) & df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df[tar_mask & df.active].groupby(by='site').mean()[val].sem(), 
                               color='k', ecolor='k', capsize=3, marker='o', markersize=ms, label='Active')
# clean pred
ax.errorbar(1, df_ind[(tar_mask | ref_mask | cat_mask) & ~df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df[tar_mask & ~df.active].groupby(by='site').mean()[val].sem(), 
                               color='grey', ecolor='grey', capsize=3, marker='o', markersize=ms)
ax.errorbar(1, df_ind[(tar_mask | ref_mask | cat_mask) & df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df[tar_mask & df.active].groupby(by='site').mean()[val].sem(), 
                               color='k', ecolor='k', capsize=3, marker='o', markersize=ms)
# clean pred
ax.errorbar(2, df_lv[(tar_mask | ref_mask | cat_mask) & ~df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df[tar_mask & ~df.active].groupby(by='site').mean()[val].sem(), 
                               color='grey', ecolor='grey', capsize=3, marker='o', markersize=ms)
ax.errorbar(2, df_lv[(tar_mask | ref_mask | cat_mask) & df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df[tar_mask & df.active].groupby(by='site').mean()[val].sem(), 
                               color='k', ecolor='k', capsize=3, marker='o', markersize=ms)
# clean pred
ax.errorbar(3, df[(tar_mask | ref_mask | cat_mask) & ~df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df[tar_mask & ~df.active].groupby(by='site').mean()[val].sem(), 
                               color='grey', ecolor='grey', capsize=3, marker='o', markersize=ms)
ax.errorbar(3, df[(tar_mask | ref_mask | cat_mask) & df.active].groupby(by='site').mean()[val].mean(),
                               yerr= df[tar_mask & df.active].groupby(by='site').mean()[val].sem(), 
                               color='k', ecolor='k', capsize=3, marker='o', markersize=ms)

ax.legend(frameon=False)
ax.set_xlim((-0.2, 3.2))
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['Clean', 'Indep', 'Indep + LV', 'Actual'], rotation=45)
ax.set_xlabel('Dataset')
ax.set_ylabel(r"$d'$")

f.tight_layout()

if savefig:
        f.savefig(DIR + '/results/figures/lvsim_dprime_results.pdf')

plt.show()