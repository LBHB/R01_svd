"""
* Noise correlations in A1 are behavior dependent (even after accounting for pupil / behavior gain changes?)
* Their state-dependence is correlated with behavioral performance, particularly in the "decision" window

Two figures:
    1) Overall change in noise correlation during 200ms decision window + correlation w/ behavior in that window
    2) Breakdown of delta noise correlation per time bin. Show where correlation strongest
"""
from settings import DIR
import scipy.stats as ss
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

df = pd.read_pickle(DIR+"results/rsc_df.pickle")
df = df[~df.active.isna() & ~df.passive.isna()]
df['diff'] = df['passive'] - df['active']
di_metric = 'DIref'
alpha = 1

# ======================== FIGURE 1 ================================
f, ax = plt.subplots(1, 2, figsize=(8, 4))

m1 = (df.batch == 307) & (df.tbin == '0.35_0.65') & ((df.pa < alpha) | (df.pp < alpha)) & (df.snr != -np.inf)
m2 = (df.batch.isin([302, 324])) & (df.tbin == '0.1_0.4') & ((df.pa < alpha) | (df.pp < alpha)) & (df.snr != -np.inf)
res = pd.concat([df[m1], df[m2]])
m3 = ~res['active'].isna() & ~res['passive'].isna()
res = res[m3]
# paired barplot, one line per site, active / passive change in noise correlation
for s in res.site.unique():
    ax[0].plot([0, 1], 
                [res[res.site==s]['active'].mean(), res[res.site==s]['passive'].mean()],
                color='grey', alpha=1, lw=0.8)
resg = res.groupby(by='site').mean()
ax[0].bar([0, 1], [resg['active'].mean(), resg['passive'].mean()], 
            yerr=[resg['active'].sem(), resg['passive'].sem()], edgecolor='k', lw=2, color='tab:orange')
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
#ax[1].scatter(resg[di_metric], resg['diff'], s=50, edgecolor='white', color='tab:orange')
sns.regplot(x=di_metric, y='diff', data=resg, ax=ax[1], color='tab:orange')
ax[1].set_xlabel('Behavior performance (DI)')
ax[1].set_ylabel(r"$\Delta r_{sc}$"+"\n(Active - Passive)")
ax[1].axhline(0, linestyle='--', color='k')
ax[1].axvline(0.5, linestyle='--', color='k')
r, p = ss.pearsonr(resg[di_metric], resg['diff'])
ax[1].set_title(r"$r$: %s, $p$: %s" % (round(r, 3), round(p, 3)))

f.tight_layout()

#f.savefig(DIR + 'pyfigures/rsc_behavior.svg')


# ===================================== FIGURE 2 ====================================
# break down correlation vs. behavior into different time windows
tbins1 = ['0.25_0.35', '0.35_0.45', '0.45_0.55', '0.55_0.65']
tbins2 = ['0_0.1', '0.1_0.2', '0.2_0.3', '0.3_0.4']
titles = ['-0.1 - 0.0 sec', '0.0 - 0.1 sec', '0.1 - 0.2 sec', '0.2 - 0.3 sec']
f, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for t1, t2, title, a in zip(tbins1, tbins2, titles, ax.flatten()):
    m1 = (df.batch == 307) & (df.tbin == t1) & ((df.pa < alpha) | (df.pp < alpha))
    m2 = (df.batch.isin([302, 324])) & (df.tbin == t2) & ((df.pa < alpha) | (df.pp < alpha))
    res = pd.concat([df[m1], df[m2]])
    resg = res.groupby(by=['snr', 'site']).mean()

    a.scatter(resg[di_metric], resg['diff'], s=50, edgecolor='white', color='tab:orange')
    a.set_xlabel('Behavior performance (DI)')
    a.set_ylabel(r"$\Delta r_{sc}$"+"\n(Active - Passive)")
    a.axhline(0, linestyle='--', color='k')
    a.axvline(0.5, linestyle='--', color='k')
    r, p = ss.pearsonr(resg[di_metric], resg['diff'])
    a.set_title(f"{title}\nr: {round(r, 3)}, pval: {round(p, 3)}")
    #a.set_ylim((-0.1, None))

f.tight_layout()

plt.show()