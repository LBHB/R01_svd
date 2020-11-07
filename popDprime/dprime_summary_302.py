"""
Active vs. passive scatter plot for three categories. 
Summary of selective effects (invariance)
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
col_per_site = False
norm = False
figsave = DIR + 'results/figures/decoding_summary302.pdf'

df = pd.read_pickle(DIR + 'results/res.pickle')
df.index = df.pair

val = 'dp_opt'
df[val] = np.sqrt(df[val])
m = 8
ms = 4

batches = [302]

tar_mask = (df.tar_tar) & (df.tdr_overall) & (~df.pca) & df.batch.isin(batches) & ~df.sim1 & (df.trials=='all')
cat_mask = (df.cat_tar) & (df.tdr_overall) & (~df.pca) & df.batch.isin(batches) & ~df.sim1 & (df.trials=='all')
ref_mask = (df.ref_ref) & (df.tdr_overall) & (~df.pca) & df.batch.isin(batches) & ~df.sim1 & (df.trials=='all')

f, ax = plt.subplots(1, 1, figsize=(2, 2), sharey=False)
ticks = [0, 0.25, 0.75, 1, 1.5, 1.75]
# normalize to mean passive across all categories
norm = pd.concat([df[ref_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val], 
            df[tar_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val], 
            df[cat_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val]]).mean()

# ==== A1 ===== 
for s in df[(df.area=='A1')].site.unique():
    p = df[ref_mask & ~df.active & (df.area=='A1') & (df.site==s)][val]
    a = df[ref_mask & df.active & (df.area=='A1') & (df.site==s)][val]
    ax.errorbar(ticks[0:2], [p.mean() / norm, a.mean() / norm], yerr=[p.sem(), a.sem()], capsize=3, 
                        color='mediumblue', markeredgecolor='k', marker='o', markersize=ms)

    p = df[tar_mask & ~df.active & (df.area=='A1') & (df.site==s)][val]
    a = df[tar_mask & df.active & (df.area=='A1') & (df.site==s)][val]
    ax.errorbar(ticks[2:4], [p.mean() / norm, a.mean() / norm], yerr=[p.sem(), a.sem()], capsize=3, 
                        color='coral', markeredgecolor='k', marker='o', markersize=ms)

    p = df[cat_mask & ~df.active & (df.area=='A1') & (df.site==s)][val]
    a = df[cat_mask & df.active & (df.area=='A1') & (df.site==s)][val]
    ax.errorbar(ticks[4:], [p.mean() / norm, a.mean() / norm], yerr=[p.sem(), a.sem()], capsize=3, 
                        color='lightgrey', markeredgecolor='k', marker='o', markersize=ms)
ax.axhline(1, linestyle='--', color='grey')
ax.set_ylabel(r"$d'$ normalized to mean passive")
ax.set_xticks(ticks)
ax.set_xticklabels(['Pas', 'Act', 'Pas', 'Act', 'Pas', 'Act'], rotation=45)

f.tight_layout()

# final ? summary plot
np.random.seed(123)
ticks = [0, 1, 2]
norm_delta = False
ms = 5
sd = 0.1
ylim = 6
f, ax = plt.subplots(1, 2, figsize=(4, 2))


# ================================== ABS DPRIME SCATTER ==============================
ax[0].scatter(df[ref_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val],
                    df[ref_mask & df.active & (df.area=='A1')].groupby(by='site').mean()[val], 
                    color='mediumblue', edgecolor='k', s=30, label='Ref vs. Ref')
ax[0].scatter(df[tar_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val],
                    df[tar_mask & df.active & (df.area=='A1')].groupby(by='site').mean()[val], 
                    color='coral', edgecolor='k', s=30, label='Tar vs. Tar')
ax[0].scatter(df[cat_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val],
                    df[cat_mask & df.active & (df.area=='A1')].groupby(by='site').mean()[val], 
                    color='lightgrey', edgecolor='k', s=30, label='Cat vs. Tar')
ax[0].legend(frameon=False)
ax[0].set_xlabel('Passive')
ax[0].set_ylabel('Active')
ax[0].set_title('A1')

m = np.max(ax[0].get_xlim() + ax[0].get_ylim())
ax[0].plot([0, m], [0, m], linestyle='--', color='grey')

# ================================== DELTA DPRIME ====================================

# ref - ref
rr_act = df[ref_mask & df.active & (df.area=='A1')][[val, 'site']].set_index('site')
rr_pass = df[ref_mask & ~df.active & (df.area=='A1')][[val, 'site']].set_index('site')
if norm_delta:
    rr_delt = (rr_act - rr_pass) / (rr_act + rr_pass)
else:
    rr_delt = rr_act - rr_pass
ax[1].scatter(np.random.normal(ticks[0], sd, len(rr_delt)),
                rr_delt, s=ms, alpha=0.1, color='mediumblue', edgecolor='none')
ax[1].errorbar(ticks[0], rr_delt.groupby(level=0).mean().mean(), yerr=rr_delt.groupby(level=0).mean().sem(), capsize=3, 
                        color='mediumblue', markeredgecolor='k', marker='o', markersize=ms, ecolor='k')
# cat - tar
ct_act = df[cat_mask & df.active & (df.area=='A1')][[val, 'site']].set_index('site')
ct_pass = df[cat_mask & ~df.active & (df.area=='A1')][[val, 'site']].set_index('site')
if norm_delta:
    ct_delt = (ct_act - ct_pass) / (ct_act + ct_pass)
else:
    ct_delt = ct_act - ct_pass
ax[1].scatter(np.random.normal(ticks[1], sd, len(ct_delt)),
                ct_delt, s=ms, alpha=0.7, color='lightgrey', edgecolor='none')
ax[1].errorbar(ticks[1], ct_delt.groupby(level=0).mean().mean(), yerr=ct_delt.groupby(level=0).mean().sem(), capsize=3, 
                        color='lightgrey', markeredgecolor='k', marker='o', markersize=ms, ecolor='k')

# tar - tar
tt_act = df[tar_mask & df.active & (df.area=='A1')][[val, 'site']].set_index('site')
tt_pass = df[tar_mask & ~df.active & (df.area=='A1')][[val, 'site']].set_index('site')
if norm_delta:
    tt_delt = (tt_act - tt_pass) / (tt_act + tt_pass)
else:
    tt_delt = tt_act - tt_pass
ax[1].scatter(np.random.normal(ticks[2], sd, len(tt_delt)),
                tt_delt, s=ms, alpha=0.7, color='coral', edgecolor='none')
ax[1].errorbar(ticks[2], tt_delt.groupby(level=0).mean().mean(), yerr=tt_delt.groupby(level=0).mean().sem(), capsize=3, 
                        color='coral', markeredgecolor='k', marker='o', markersize=ms, ecolor='k')

ax[1].axhline(0, linestyle='--', color='grey', zorder=0)
ax[1].set_ylabel(r"$\Delta d'$")
ax[1].set_ylim((None, ylim))

f.tight_layout()

if savefig:
    f.savefig(figsave)

plt.show()
