from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems_lbhb.tin_helpers as thelp
from sklearn.decomposition import PCA
from itertools import combinations
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': 6})
np.random.seed(123)

batch = 324
site = 'CRD016c'
options = {'rasterfs': 10, 'resp': True, 'pupil': True}

manager = BAPHYExperiment(batch=batch, siteid=site[:7])
rec = manager.get_recording(**options)
rec['resp'] = rec['resp'].rasterize()

start = int(0.1 * options['rasterfs'])
end = int(0.3 * options['rasterfs'])

rec = rec.and_mask(['HIT_TRIAL', 'MISS_TRIAL', 'CORRECT_REJECT_TRIAL', 'INCORRECT_HIT_TRIAL'])
rec = rec.apply_mask(reset_epochs=True)

# look for HIT / MISS dependent changes in PC / dDR space and/or early/late changes
targets = thelp.sort_targets([t for t in rec.epochs.name.unique() if 'TAR_' in t])
catch = [t for t in rec.epochs.name.unique() if 'CAT_' in t]
ref = thelp.sort_refs([t for t in rec.epochs.name.unique() if 'STIM_' in t])

BwG, gR = thelp.make_tbp_colormaps(ref_stims=ref, tar_stims=catch+targets, use_tar_freq_idx=0)

# project targets / catches onto first two PCs, marker type indicates trial type
ct_resp = rec['resp'].extract_epochs(catch+targets)
pup = rec['pupil'].extract_epochs(catch+targets)
hit_mask = rec['resp'].epoch_to_signal('HIT_TRIAL').extract_epochs(catch+targets)
miss_mask = rec['resp'].epoch_to_signal('MISS_TRIAL').extract_epochs(catch+targets)
cr_mask = rec['resp'].epoch_to_signal('CORRECT_REJECT_TRIAL').extract_epochs(catch+targets)
inc_mask = rec['resp'].epoch_to_signal('INCORRECT_HIT_TRIAL').extract_epochs(catch+targets)
pc_resp = np.stack([v[:, :, start:end].mean(axis=(0, -1)) for k, v in ct_resp.items()])

pca = PCA()
pca.fit(pc_resp)

f, ax = plt.subplots(1, 1, figsize=(5, 5))

for i, k in enumerate(ct_resp.keys()):
    x = ct_resp[k][:, :, start:end].mean(axis=-1).dot(pca.components_[[0], :].T).squeeze()
    y = ct_resp[k][:, :, start:end].mean(axis=-1).dot(pca.components_[[1], :].T).squeeze()
    hm = hit_mask[k][:, 0, 0]
    mm = miss_mask[k][:, 0, 0]
    crm = cr_mask[k][:, 0, 0]
    icm = inc_mask[k][:, 0, 0]

    if hm.sum() > 1:
        ax.scatter(x[hm], y[hm], color=gR(i), marker='o')
    if mm.sum() > 1:
        ax.scatter(x[mm], y[mm], color=gR(i), marker='^')
    if crm.sum() > 1:
        ax.scatter(x[crm], y[crm], color=gR(i), marker='*')
    if icm.sum() > 1:
        ax.scatter(x[icm], y[icm], color=gR(i), facecolor='none', marker='o')

ax.set_xlabel(r"$PC_1$")
ax.set_ylabel(r"$PC_2$")

f.tight_layout()

# for each cat / tar pair, plot distribution on first two PCs for hits/misses, and cr/FA
tar_cat_pairs = list(itertools.product(catch, targets)) 
for i, tc in enumerate(tar_cat_pairs):
    # just scatter plot, color by time
    f, ax = plt.subplots(2, 1, figsize=(2, 4))
    cat = ct_resp[tc[0]][:, :, start:end].mean(axis=-1).dot(pca.components_[0:2, :].T).squeeze()
    tar = ct_resp[tc[1]][:, :, start:end].mean(axis=-1).dot(pca.components_[0:2, :].T).squeeze()

    ax[0].scatter(cat[:, 0], cat[:, 1], c=range(0, cat.shape[0]), cmap='Greys')
    ax[0].scatter(tar[:, 0], tar[:, 1], c=range(0, tar.shape[0]), cmap='Reds')

    ax[0].set_xlabel(r"$PC_1$")
    ax[0].set_ylabel(r"$PC_2$")
    ax[0].set_title(tc[0]+'\n'+tc[1], fontsize=6)

    # just scatter plot, color by pupil
    cat = ct_resp[tc[0]][:, :, start:end].mean(axis=-1).dot(pca.components_[0:2, :].T).squeeze()
    tar = ct_resp[tc[1]][:, :, start:end].mean(axis=-1).dot(pca.components_[0:2, :].T).squeeze()

    ax[1].scatter(cat[:, 0], cat[:, 1], c=pup[tc[0]][:, 0, start:end].mean(axis=-1), cmap='Greys')
    ax[1].scatter(tar[:, 0], tar[:, 1], c=pup[tc[1]][:, 0, start:end].mean(axis=-1), cmap='Reds')

    ax[1].set_xlabel(r"$PC_1$")
    ax[1].set_ylabel(r"$PC_2$")
   
    f.tight_layout()

    hm = hit_mask[tc[1]][:, 0, 0]
    mm = miss_mask[tc[1]][:, 0, 0]
    crm = cr_mask[tc[0]][:, 0, 0]
    icm = inc_mask[tc[0]][:, 0, 0]

    #if hm.sum() > 1:
    #    ax.scatter(tar[:, 0][hm], tar[:, 1][hm], color='red', marker='o')
    #if mm.sum() > 1:
    #    ax.scatter(tar[:, 0][mm], tar[:, 1][mm], color='coral', facecolor='none', marker='o')
    #if crm.sum() > 1:
    #    ax.scatter(cat[:, 0][crm], cat[:, 1][crm], color='blue', marker='o')
    #if icm.sum() > 1:
    #    ax.scatter(cat[:, 0][icm], cat[:, 1][icm], color='lightblue', facecolor='none', marker='o')

    #ax.set_title(tc)
    #ax.set_xlabel(r"$PC_1$")
    #ax.set_ylabel(r"$PC_2$")

    #f.tight_layout()

    # plot on FA axis
    f, ax = plt.subplots(2, 1, figsize=(3, 3))
    dU = ct_resp[tc[0]][:, :, start:end].mean(axis=(0, -1)) - ct_resp[tc[1]][:, :, start:end].mean(axis=(0, -1))
    dU /= np.linalg.norm(dU)

    dU = ct_resp[tc[0]][:, :, start:end].mean(axis=-1)[crm].mean(axis=0) - ct_resp[tc[0]][:, :, start:end].mean(axis=-1)[icm].mean(axis=0)
    dU /= np.linalg.norm(dU)

    cat = ct_resp[tc[0]][:, :, start:end].mean(axis=-1).dot(dU[:, np.newaxis]).squeeze()
    tar = ct_resp[tc[1]][:, :, start:end].mean(axis=-1).dot(dU[:, np.newaxis]).squeeze()

    bins = np.linspace(np.min([np.min(cat), np.min(tar)]), np.max([np.max(cat), np.max(tar)]), 10)
    if hm.sum() > 1:
        ax[0].hist(tar[hm], bins=bins, color='red', histtype='step', lw=2)
    if mm.sum() > 1:
        ax[1].hist(tar[mm], bins=bins, color='coral', histtype='step', lw=2)
    if crm.sum() > 1:
        ax[0].hist(cat[crm], bins=bins, color='blue', histtype='step', lw=2)
    if icm.sum() > 1:
        ax[1].hist(cat[icm], bins=bins, color='lightblue', histtype='step', lw=2)
    ax[0].set_title(tc, fontsize=6)

plt.show()