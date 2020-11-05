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

savefig = False
col_per_site = False
norm = False
figsave = DIR + 'results/figures/decodng_summary.pdf'

df = pd.read_pickle(DIR + 'results/res.pickle')
df.index = df.pair

val = 'dp_opt'
df[val] = np.sqrt(df[val])
m = 8
ms = 30

tar_mask = (df.tar_tar) & (df.tdr_overall) & (~df.pca) & df.batch.isin([324, 325]) & (df.f1 == df.f2) & ~df.sim1
cat_mask = (df.cat_tar) & (df.tdr_overall) & (~df.pca) & df.batch.isin([324, 325]) & (df.f1 == df.f2) & ~df.sim1
ref_mask = (df.ref_ref) & (df.tdr_overall) & (~df.pca) & df.batch.isin([324, 325]) & ~df.sim1

f, ax = plt.subplots(2, 2, figsize=(4, 4))

# ==== A1 ===== 
ax[0, 0].scatter(y=df[tar_mask & df.active & (df.area=='A1')].groupby(by='site').mean()[val],
            x=df[tar_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val], s=ms, edgecolor='k', color='coral', label='Tar vs. Tar')
ax[0, 0].scatter(y=df[cat_mask & df.active & (df.area=='A1')].groupby(by='site').mean()[val],
            x=df[cat_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val], s=ms, edgecolor='k', color='lightgrey', label='Tar vs. Cat')
ax[0, 0].scatter(y=df[ref_mask & df.active & (df.area=='A1')].groupby(by='site').mean()[val],
            x=df[ref_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val], s=ms, edgecolor='k', color='mediumblue', label='Ref vs. Ref')
ax[0, 0].set_xlabel(r"$d'$ Active")
ax[0, 0].set_ylabel(r"$d'$ Passive")
ax[0, 0].plot([0, m], [0, m], '--', color='grey', lw=2)
ax[0, 0].set_title('A1')
ax[0, 0].legend(frameon=False)

# ==== PEG ===== 
ax[0, 1].scatter(y=df[tar_mask & df.active & (df.area=='PEG')].groupby(by='site').mean()[val],
            x=df[tar_mask & ~df.active & (df.area=='PEG')].groupby(by='site').mean()[val], s=ms, edgecolor='k', color='coral')
ax[0, 1].scatter(y=df[cat_mask & df.active & (df.area=='PEG')].groupby(by='site').mean()[val],
            x=df[cat_mask & ~df.active & (df.area=='PEG')].groupby(by='site').mean()[val], s=ms, edgecolor='k', color='lightgrey')
ax[0, 1].scatter(y=df[ref_mask & df.active & (df.area=='PEG')].groupby(by='site').mean()[val],
            x=df[ref_mask & ~df.active & (df.area=='PEG')].groupby(by='site').mean()[val], s=ms, edgecolor='k', color='mediumblue')
ax[0, 1].set_xlabel(r"$d'$ Active")
ax[0, 1].set_ylabel(r"$d'$ Passive")
ax[0, 1].plot([0, m], [0, m], '--', color='grey', lw=2)
ax[0, 1].set_title('PEG')

# normalize changes and look within site
tt_act = df[tar_mask & df.active & (df.area=='A1')][[val, 'site']].set_index('site')
tt_pass = df[tar_mask & ~df.active & (df.area=='A1')][[val, 'site']].set_index('site')
if norm:
    n = (tt_act + tt_pass)
else:
    n = 1
tt_delta = ((tt_act - tt_pass) / n).groupby(level=0).mean()
tt_sem = ((tt_act - tt_pass) / n).groupby(level=0).sem()

ct_act = df[cat_mask & df.active & (df.area=='A1')][[val, 'site']].set_index('site')
ct_pass = df[cat_mask & ~df.active & (df.area=='A1')][[val, 'site']].set_index('site')
if norm:
    n = (ct_act + ct_pass)
else:
    n = 1
ct_delta = ((ct_act - ct_pass) / n).groupby(level=0).mean()
ct_sem = ((ct_act - ct_pass) / n).groupby(level=0).sem()

rr_act = df[ref_mask & df.active & (df.area=='A1')][[val, 'site']].set_index('site')
rr_pass = df[ref_mask & ~df.active & (df.area=='A1')][[val, 'site']].set_index('site')
if norm:
    n = (rr_act + rr_pass)
else:
    n = 1
rr_delta = ((rr_act - rr_pass) / n).groupby(level=0).mean()
rr_sem = ((rr_act - rr_pass) / n).groupby(level=0).sem()

colors = plt.get_cmap('jet', len(ct_delta.index))
cells = pd.concat([nd.get_batch_cells(324).cellid, nd.get_batch_cells(302).cellid])
for i, s in enumerate(list(set(ct_delta.index).intersection(set(tt_delta.index)))):
    ncells = len([c for c in cells if s in c])
    
    if col_per_site:
        c = colors(i)
        lab = s + f' ({ncells} cells)'
    else:
        lab = None
        c = 'k'
    ax[1, 0].errorbar([0, 1, 2], [rr_delta.loc[s].values[0], tt_delta.loc[s].values[0], ct_delta.loc[s].values[0]], \
        yerr=[rr_sem.loc[s].values[0], tt_sem.loc[s].values[0], ct_sem.loc[s].values[0]], capsize=3, color=c, label=lab, marker='o')

#leg = ax[1, 0].legend(frameon=False, handlelength=0)
#for line, text in zip(leg.get_lines(), leg.get_texts()):
#    text.set_color(line.get_color())
ax[1, 0].set_xticks([0, 1, 2])
ax[1, 0].axhline(0, linestyle='--', color='grey', lw=2)
ax[1, 0].set_xticklabels(['Ref vs. Ref,', 'Tar. vs. Tar', 'Cat vs. Tar'], rotation=45)
ax[1, 0].set_ylabel(r"$\Delta d'$")

# normalize changes and look within site
tt_act = df[tar_mask & df.active & (df.area=='PEG')][[val, 'site']].set_index('site')
tt_pass = df[tar_mask & ~df.active & (df.area=='PEG')][[val, 'site']].set_index('site')
if norm:
    n = (tt_act + tt_pass)
else:
    n = 1
tt_delta = ((tt_act - tt_pass) / n).groupby(level=0).mean()
tt_sem = ((tt_act - tt_pass) / n).groupby(level=0).sem()

ct_act = df[cat_mask & df.active & (df.area=='PEG')][[val, 'site']].set_index('site')
ct_pass = df[cat_mask & ~df.active & (df.area=='PEG')][[val, 'site']].set_index('site')
if norm:
    n = (ct_act + ct_pass)
else:
    n = 1
ct_delta = ((ct_act - ct_pass) / n).groupby(level=0).mean()
ct_sem = ((ct_act - ct_pass) / n).groupby(level=0).sem()

rr_act = df[ref_mask & df.active & (df.area=='PEG')][[val, 'site']].set_index('site')
rr_pass = df[ref_mask & ~df.active & (df.area=='PEG')][[val, 'site']].set_index('site')
if norm:
    n = (rr_act + rr_pass)
else:
    n = 1
rr_delta = ((rr_act - rr_pass) / n).groupby(level=0).mean()
rr_sem = ((rr_act - rr_pass) / n).groupby(level=0).sem()

colors = plt.get_cmap('jet', len(ct_delta.index))
cells = nd.get_batch_cells(325).cellid
for i, s in enumerate(list(set(ct_delta.index).intersection(set(tt_delta.index)))):
    ncells = len([c for c in cells if s in c])
    if col_per_site:
        c = colors(i)
        lab = s + f' ({ncells} cells)'
    else:
        lab = None
        c = 'k'
    ax[1, 1].errorbar([0, 1, 2], [rr_delta.loc[s].values[0], tt_delta.loc[s].values[0], ct_delta.loc[s].values[0]], \
        yerr=[rr_sem.loc[s].values[0], tt_sem.loc[s].values[0], ct_sem.loc[s].values[0]], capsize=3, color=c, label=lab, marker='o')

leg = ax[1, 1].legend(frameon=False, handlelength=0)
for line, text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())
ax[1, 1].set_xticks([0, 1, 2])
ax[1, 1].axhline(0, linestyle='--', color='grey', lw=2)
ax[1, 1].set_xticklabels(['Ref vs. Ref', 'Tar. vs. Tar', 'Cat vs. Tar'], rotation=45)
ax[1, 1].set_ylabel(r"$\Delta d'$")

ma = np.max(ax[1, 0].get_ylim() + ax[1, 1].get_ylim())
mi = np.min(ax[1, 0].get_ylim() + ax[1, 1].get_ylim())
ax[1, 0].set_ylim((mi, ma))
ax[1, 1].set_ylim((mi, ma))
f.tight_layout()

if savefig:
    f.savefig(figsave)

plt.show()