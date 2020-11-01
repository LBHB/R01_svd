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
m = 20
ms = 4

tar_mask = (df.tar_tar) & (df.tdr_overall==True) & (~df.pca) & df.batch.isin([324, 325]) & (df.f1 == df.f2) 
cat_mask = (df.cat_tar) & (df.tdr_overall==True) & (~df.pca) & df.batch.isin([324, 325]) & (df.f1 == df.f2) 
ref_mask = (df.ref_ref) & (df.tdr_overall==True) & (~df.pca) & df.batch.isin([324, 325])

# compute delte dprime for simulations and for raw data, for each category, for each area


f, ax = plt.subplots(2, 1, figsize=(2, 4))

# ==== A1 ===== 
ax[0].scatter(y=(df[tar_mask & df.active & (df.area=='A1')].groupby(by='site').mean()[val]) - (df[tar_mask & ~df.active & (df.area=='A1')][val]),
            x=(df[tar_mask & df.active & (df.area=='A1')][val]) - (df[tar_mask & ~df.active & (df.area=='A1')][val]), s=ms, color=cmap['tar_tar'], label='Tar vs. Tar')
ax[0].scatter(y=df[cat_mask & df.active & (df.area=='A1')][val],
            x=df[cat_mask & ~df.active & (df.area=='A1')][val], s=ms, color=cmap['cat_tar'], label='Tar vs. Cat')
ax[0].scatter(y=df[ref_mask & df.active & (df.area=='A1')][val],
            x=df[ref_mask & ~df.active & (df.area=='A1')][val], s=ms, color=cmap['ref_ref'], label='Ref vs. Ref', zorder=-3)
ax[0].set_xlabel(r"$d'$ Active")
ax[0].set_ylabel(r"$d'$ Passive")
ax[0].plot([0, m], [0, m], '--', color='grey', lw=2)
ax[0].set_title('A1')
ax[0].legend(frameon=False)

# ==== PEG ===== 
ax[1].scatter(y=df[tar_mask & df.active & (df.area=='PEG')][val],
            x=df[tar_mask & ~df.active & (df.area=='PEG')][val], s=ms, color=cmap['tar_tar'])
ax[1].scatter(y=df[cat_mask & df.active & (df.area=='PEG')][val],
            x=df[cat_mask & ~df.active & (df.area=='PEG')][val], s=ms, color=cmap['cat_tar'])
ax[1].scatter(y=df[ref_mask & df.active & (df.area=='PEG')][val],
            x=df[ref_mask & ~df.active & (df.area=='PEG')][val], s=ms, color=cmap['ref_ref'], zorder=-3)
ax[1].set_xlabel(r"$d'$ Active")
ax[1].set_ylabel(r"$d'$ Passive")
ax[1].plot([0, m], [0, m], '--', color='grey', lw=2)
ax[1].set_title('PEG')

f.tight_layout()

plt.show()