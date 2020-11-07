"""
Look at variance in decoding space as function of:
    passive trials, early trials, late trials, behavior performance
"""
from settings import DIR
import nems.db as nd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 6

savefig = True
figsave1 = DIR + 'results/figures/variance_task_block.pdf'
figsave2 = DIR + 'results/figures/variance_performance.pdf'

df = pd.read_pickle(DIR + 'results/res.pickle')
df.index = df.pair
df['var'] = df['evals'].apply(lambda x: sum(x))

amask = (df.cat_tar) & (df.tdr_overall) & (~df.pca) & df.batch.isin([324, 325]) & ~df.sim1 & (df.trials=='all')
emask = (df.cat_tar) & (df.tdr_overall) & (~df.pca) & df.batch.isin([324, 325]) & ~df.sim1 & (df.trials=='early')
lmask = (df.cat_tar) & (df.tdr_overall) & (~df.pca) & df.batch.isin([324, 325]) & ~df.sim1 & (df.trials=='late')

# plot passive / early / late total variance in dDR space, for hard / easy discrimination
ms = 5
f, ax = plt.subplots(1, 2, figsize=(2, 2), sharey=True)

pdata = df[amask & ~df.active & (df.snr1==-5)]
edata = df[emask & (df.snr1==-5)]
ldata = df[lmask & (df.snr1==-5)]
ax[0].plot([[0] * pdata.shape[0], [1] * pdata.shape[0], [2] * pdata.shape[0]],
                [pdata['var'], edata['var'], ldata['var']], color='lightgrey', zorder=0)
ax[0].errorbar([0, 1, 2], 
                [pdata['var'].mean(), edata['var'].mean(), ldata['var'].mean()],
                yerr=[pdata['var'].sem(), edata['var'].sem(), ldata['var'].sem()], 
                capsize=3, color='k', marker='o', zorder=1, markersize=ms)
ax[0].set_xlim((-0.5, 2.5))
ax[0].set_xticks([0, 1, 2])
ax[0].set_xticklabels(['Passive', 'Early', 'Late'], rotation=45)
pvalue1 = round(ss.wilcoxon(pdata['var'], edata['var']).pvalue, 3)
pvalue2 = round(ss.wilcoxon(edata['var'], ldata['var']).pvalue, 3)
ax[0].set_title(f'Low SNR \n p: {pvalue1}, {pvalue2}')
ax[0].set_ylabel(r"$dDR$ space variance")

pdata = df[amask & ~df.active & ((df.snr1==np.inf) | (df.snr1==0))]
edata = df[emask & ((df.snr1==np.inf) | (df.snr1==0))]
ldata = df[lmask & ((df.snr1==np.inf) | (df.snr1==0))]
ax[1].plot([[0] * pdata.shape[0], [1] * pdata.shape[0], [2] * pdata.shape[0]],
                [pdata['var'], edata['var'], ldata['var']], color='lightgrey', zorder=0)
ax[1].errorbar([0, 1, 2], 
                [pdata['var'].mean(), edata['var'].mean(), ldata['var'].mean()],
                yerr=[pdata['var'].sem(), edata['var'].sem(), ldata['var'].sem()], 
                capsize=3, color='k', marker='o', zorder=1, markersize=ms)
ax[1].set_xlim((-0.5, 2.5))
ax[1].set_xticks([0, 1, 2])
ax[1].set_xticklabels(['Passive', 'Early', 'Late'], rotation=45)
pvalue1 = round(ss.wilcoxon(pdata['var'], edata['var']).pvalue, 3)
pvalue2 = round(ss.wilcoxon(edata['var'], ldata['var']).pvalue, 3)
ax[1].set_title(f'High SNR \n p: {pvalue1}, {pvalue2}')


f.tight_layout()

if savefig:
    f.savefig(figsave1)


amask = (df.aref_tar) & (df.tdr_overall) & (~df.pca) & df.batch.isin([302, 307, 324, 325]) & ~df.sim1 & (df.trials=='all')

f, ax = plt.subplots(1, 1, figsize=(2, 2))

var_diff = (df[amask & df.active]['var'] - df[amask & ~df.active]['var'])# / (df[amask & df.active]['var'] + df[amask & ~df.active]['var'])
perf = df[amask & df.active]['DI']

g = sns.regplot(x=perf, y=var_diff, ax=ax)
ax.set_xlabel('Behavior performance (DI)')
ax.set_ylabel(r"$\Delta dDR$ variance")
r, p = ss.pearsonr(perf, var_diff)
ax.set_title(f"r: {round(r, 3)}, p: {round(p, 3)}")
ax.axhline(0, linestyle='--', color='grey')
ax.axvline(0.5, linestyle='--', color='grey')

f.tight_layout()

if savefig:
    f.savefig(figsave2)

plt.show()