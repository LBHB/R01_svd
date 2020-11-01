"""
Two figures:
    1) Overall change in noise correlation during 200ms decision window
    2) correlation w/ behavior in that window

* color results by batch
"""
from settings import DIR
import scipy.stats as ss
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 6

df = pd.read_pickle(DIR+"results/rsc_df.pickle")
df = df[~df.active.isna() & ~df.passive.isna()]
df['diff'] = df['passive'] - df['active']
di_metric = 'DIref'
alpha = 1
ms = 5

cmap = {
    302: 'tab:blue',
    307: 'tab:orange',
    324: 'tab:green',
    325: 'tab:red',
}

# ======================== FIGURE 1 ================================
f, ax = plt.subplots(1, 2, figsize=(4, 2))

m1 = (df.batch == 307) & (df.tbin == '0.35_0.65') & ((df.pa < alpha) | (df.pp < alpha)) & (df.snr != -np.inf)
m2 = (df.batch.isin([302, 324, 325])) & (df.tbin == '0.1_0.4') & ((df.pa < alpha) | (df.pp < alpha)) & (df.snr != -np.inf)
res = pd.concat([df[m1], df[m2]])
m3 = ~res['active'].isna() & ~res['passive'].isna()
res = res[m3]
# paired barplot, one line per site, active / passive change in noise correlation
for s in res.site.unique():
    batch = res[res.site==s]['batch'].unique()[0]
    color = cmap[batch]
    ax[0].plot([0, 1], 
                [res[res.site==s]['active'].mean(), res[res.site==s]['passive'].mean()],
                color=color, alpha=1, lw=0.8)
resg = res.groupby(by='site').mean()
ax[0].set_xlim((-2, 3))
if alpha < 1:
    ax[0].set_ylim((-0.025, 0.25))
else:
    ax[0].set_ylim((-0.02, 0.12))
ax[0].axhline(0, linestyle='--', color='k')
ax[0].set_xticks([0, 1])
ax[0].set_xticklabels(['Active', 'Passive'], rotation=45)
ax[0].set_xlabel('Behavior State')
ax[0].set_ylabel(r'Noise Correlation ($r_{sc}$)')

# correlation with overall behavior
resg = res.groupby(by=['snr', 'f', 'site']).mean()
for s in res.site.unique():
    batch = res[res.site==s]['batch'].unique()[0]
    color = cmap[batch]
    ax[1].scatter(resg.loc[pd.IndexSlice[:, :, s], di_metric], resg.loc[pd.IndexSlice[:, :, s], 'diff'], s=ms, color=color)

sns.regplot(x=di_metric, y='diff', data=resg, ax=ax[1], color='grey', marker='')

ax[1].set_xlabel('Behavior performance (DI)')
ax[1].set_ylabel(r"$\Delta r_{sc}$"+"\n(Active - Passive)")
ax[1].axhline(0, linestyle='--', color='k')
ax[1].axvline(0.5, linestyle='--', color='k')

custom_lines = [Line2D([0], [0], color=cmap[302], lw=1),
                Line2D([0], [0], color=cmap[307], lw=1),
                Line2D([0], [0], color=cmap[324], lw=1),
                Line2D([0], [0], color=cmap[325], lw=1)]
ax[1].legend(custom_lines, ['302', '307', '324', '325'], frameon=False)

r, p = ss.pearsonr(resg[di_metric], resg['diff'])
ax[1].set_title(r"$r$: %s, $p$: %s" % (round(r, 3), round(p, 3)))

f.tight_layout()

f.savefig(DIR + '/results/figures/rsc_behavior.pdf')

plt.show()