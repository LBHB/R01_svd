"""
Similar to invariance.py for batch 324/325
but here we're talking about frequency discrimination within/outside Reward category
"""

"""
tar vs. cat, tar vs. tar, ref vs. ref decoding in A1 / PEG.
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
mpl.rcParams['font.size'] = 14

savefig = False
figsave = DIR + 'results/figures/invariance.pdf'

df = pd.read_pickle(DIR + 'results/res.pickle')
df.index = df.pair

val = 'dp_opt'
df[val] = np.sqrt(df[val])
m = 14

tar_mask = (df.tar_tar) & (df.tdr_overall==True) & (~df.pca) & df.batch.isin([302]) 
cat_mask = (df.cat_tar) & (df.tdr_overall==True) & (~df.pca) & df.batch.isin([302])
ref_mask = (df.ref_ref) & (df.tdr_overall==True) & (~df.pca) & df.batch.isin([302])

f, ax = plt.subplots(1, 2, figsize=(10, 5))

# ==== A1 ===== 
ax[0].scatter(df[tar_mask & df.active & (df.area=='A1')].groupby(by='site').mean()[val],
            df[tar_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val], s=80, edgecolor='k', label='Rew vs. Rew')
ax[0].scatter(df[cat_mask & df.active & (df.area=='A1')].groupby(by='site').mean()[val],
            df[cat_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val], s=80, edgecolor='k', label='Rew vs. N. Rew')
ax[0].scatter(df[ref_mask & df.active & (df.area=='A1')].groupby(by='site').mean()[val],
            df[ref_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val], s=80, edgecolor='k', label='Ref vs. Ref')
ax[0].set_xlabel(r"$d'$ Active")
ax[0].set_ylabel(r"$d'$ Passive")
ax[0].plot([0, m], [0, m], '--', color='grey', lw=2)
ax[0].set_title('A1 (batch 302)')
ax[0].legend(frameon=False)

# normalize changes and look within site
tt_act = df[tar_mask & df.active & (df.area=='A1')][[val, 'site']].set_index('site')
tt_pass = df[tar_mask & ~df.active & (df.area=='A1')][[val, 'site']].set_index('site')
tt_delta = ((tt_act - tt_pass) / (tt_act + tt_pass)).groupby(level=0).mean()
tt_sem = ((tt_act - tt_pass) / (tt_act + tt_pass)).groupby(level=0).sem()
ct_act = df[cat_mask & df.active & (df.area=='A1')][[val, 'site']].set_index('site')
ct_pass = df[cat_mask & ~df.active & (df.area=='A1')][[val, 'site']].set_index('site')
ct_delta = ((ct_act - ct_pass) / (ct_act + ct_pass)).groupby(level=0).mean()
ct_sem = ((ct_act - ct_pass) / (ct_act + ct_pass)).groupby(level=0).sem()
rr_act = df[ref_mask & df.active & (df.area=='A1')][[val, 'site']].set_index('site')
rr_pass = df[ref_mask & ~df.active & (df.area=='A1')][[val, 'site']].set_index('site')
rr_delta = ((rr_act - rr_pass) / (rr_act + rr_pass)).groupby(level=0).mean()
rr_sem = ((rr_act - rr_pass) / (rr_act + rr_pass)).groupby(level=0).sem()

colors = plt.get_cmap('jet', len(ct_delta.index))
cells = [c for c in nd.get_batch_cells(302).cellid if 'gus' not in c]
for i, s in enumerate(list(set(ct_delta.index).intersection(set(tt_delta.index)))):
    ncells = len([c for c in cells if (s[:7] in c) & (int(c.split('-')[1]) in range(int(s.split('.e')[1].split(':')[0]), int(s.split('.e')[1].split(':')[1])))])
    lab = s + f' ({ncells} cells)'
    ax[1].errorbar([0, 1, 2], [rr_delta.loc[s].values[0], tt_delta.loc[s].values[0], ct_delta.loc[s].values[0]], \
        yerr=[rr_sem.loc[s].values[0], tt_sem.loc[s].values[0], ct_sem.loc[s].values[0]], capsize=3, color=colors(i), label=lab, marker='o')

leg = ax[1].legend(frameon=False, handlelength=0, fontsize=8)
for line, text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())
ax[1].set_xticks([0, 1, 2])
ax[1].axhline(0, linestyle='--', color='grey', lw=2)
ax[1].set_xticklabels(['Ref vs. Ref', 'Rew vs. Rew', 'N. Rew vs. Rew'], rotation=45)
ax[1].set_ylabel(r"$\Delta d'$")

f.tight_layout()

if savefig:
    f.savefig(figsave)

plt.show()