"""
Analyses catch responses as function of 
    1) correct reject / incorrect hit (FA)
    2) early / late trials
* Compare if the catch response is closer / further from each target 
    based on these conditions
* Does resp variance change early / late?
"""
from charlieTools.decoding import compute_dprime
from charlieTools.dim_reduction import TDR

import nems.db as nd
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
mpl.rcParams.update({'font.size': 12})
np.random.seed(123)

batches = [324, 325]
options = {'rasterfs': 10, 'resp': True, 'pupil': True}
start = int(0.1 * options['rasterfs'])
end = int(0.3 * options['rasterfs'])

df = pd.DataFrame()
df2 = pd.DataFrame()
for batch in batches:
    sites = np.unique([c[:7] for c in nd.get_batch_cells(batch).cellid])
    sites = [s for s in sites if (s!='CRD013b') & ('gus' not in s) & (s!='ARM007c')]  
    for site in sites:
        manager = BAPHYExperiment(batch=batch, siteid=site[:7])
        rec = manager.get_recording(**options)
        rec['resp'] = rec['resp'].rasterize()


        rec = rec.and_mask(['HIT_TRIAL', 'MISS_TRIAL', 'CORRECT_REJECT_TRIAL', 'INCORRECT_HIT_TRIAL'])
        rec = rec.apply_mask(reset_epochs=True)

        # get sound epoch names
        targets = thelp.sort_targets([t for t in rec.epochs.name.unique() if 'TAR_' in t])
        catch = [t for t in rec.epochs.name.unique() if 'CAT_' in t]
        ref = thelp.sort_refs([t for t in rec.epochs.name.unique() if 'STIM_' in t])

        BwG, gR = thelp.make_tbp_colormaps(ref_stims=ref, tar_stims=catch+targets, use_tar_freq_idx=0)

        ct_resp = rec['resp'].extract_epochs(catch+targets)
        pup = rec['pupil'].extract_epochs(catch+targets)
        hit_mask = rec['resp'].epoch_to_signal('HIT_TRIAL').extract_epochs(catch+targets)
        miss_mask = rec['resp'].epoch_to_signal('MISS_TRIAL').extract_epochs(catch+targets)
        cr_mask = rec['resp'].epoch_to_signal('CORRECT_REJECT_TRIAL').extract_epochs(catch+targets)
        inc_mask = rec['resp'].epoch_to_signal('INCORRECT_HIT_TRIAL').extract_epochs(catch+targets)


        # define the "overall" PC and dDR spaces
        cat = np.vstack([v[:, :, start:end].mean(axis=-1) for (k, v) in ct_resp.items() if 'CAT_' in k])
        tar = np.vstack([v[:, :, start:end].mean(axis=-1) for (k, v) in ct_resp.items() if 'TAR_' in k])
        uall = np.vstack([v[:, :, start:end].mean(axis=-1).mean(axis=0) for (k, v) in ct_resp.items()])
        pca = PCA(n_components=2)
        pca.fit(uall)
        pc_axes = pca.components_
        tdr = TDR()
        tdr.fit(cat, tar)
        tdr_axes = tdr.weights

        # for each cat / tar pair, compare correct rejects to FAs
        # determine 
        #    if FAs look more/less similar to target than CR
        #    do this by computing d' in a few diff spaces
        #         overall tar/catch PC space
        #         overall tar/catch dDR space
        #         specific tar/catch pair dDR space
        #         specific cr/fa axis
        tar_cat_pairs = list(itertools.product(catch, targets)) 
        for i, tc in enumerate(tar_cat_pairs):
            # get pair dDR weights 
            c = ct_resp[tc[0]][:, :, start:end].mean(axis=-1)
            t = ct_resp[tc[1]][:, :, start:end].mean(axis=-1)
            tdr = TDR()
            tdr.fit(c, t)
            ptdr_axes = tdr.weights
            # get cr/fa axis for this catch
            cr = c[cr_mask[tc[0]][:,0,0], :]
            fa = c[inc_mask[tc[0]][:,0,0], :]
            choice_axes = cr.mean(axis=0) - fa.mean(axis=0, keepdims=True)
            choice_axes /= np.linalg.norm(choice_axes)

            # get some relevant pair information
            snr1 = thelp.get_snrs([tc[0]])[0]
            snr2 = thelp.get_snrs([tc[1]])[0]
            f1 = thelp.get_tar_freqs([tc[0]])[0]
            f2 = thelp.get_tar_freqs([tc[1]])[0]
            pstr = tc[0] + '_' + tc[1]

            # for each set of axes, compute dprime / other metrics
            for ax, category in zip([pc_axes, tdr_axes, ptdr_axes, choice_axes], ['pca', 'tdr_overall', 'tdr_pair', 'choice']): 
                # define decoding axis using all data, but evenly random sample cr / fa to avoid bias
                cr_idx = np.argwhere(cr_mask[tc[0]][:,0,0]).squeeze()
                fa_idx = np.argwhere(inc_mask[tc[0]][:,0,0]).squeeze()
                min_trials = np.min([len(cr_idx), len(fa_idx)])
                skip = False
                if min_trials < 4:
                    # skip
                    skip = True
                if not skip:
                    cidx = np.concatenate((np.random.choice(cr_idx, min_trials, replace=False), 
                                            np.random.choice(fa_idx, min_trials, replace=False)))

                    dp, wopt, evals, evecs, evec_sim, dU = compute_dprime(c[cidx].dot(ax.T).T, t.dot(ax.T).T)
                    dp_diag, _, _, _, _, _ = compute_dprime(c.dot(ax.T).T, t.dot(ax.T).T, diag=True)
                    df = df.append(pd.DataFrame(data=[dp, wopt, evecs, evals, evec_sim, dU, dp_diag, category, 'all',
                                snr1, snr2, f1, f2, ax, pstr, site, batch],
                                index=['dp_opt', 'wopt', 'evecs', 'evals', 'evec_sim', 'dU', 'dp_diag', 'dr_space', 'trials',
                                    'snr1', 'snr2', 'f1', 'f2', 'dr_weights', 'pair', 'site', 'batch']).T)
                    # CR trials (all targets)
                    dp, _, evals, evecs, evec_sim, dU = compute_dprime(cr.dot(ax.T).T, t.dot(ax.T).T, wopt=wopt)
                    dp_diag, _, _, _, _, _ = compute_dprime(cr.dot(ax.T).T, t.dot(ax.T).T, diag=True)
                    df = df.append(pd.DataFrame(data=[dp, wopt, evecs, evals, evec_sim, dU, dp_diag, category, 'cr',
                                snr1, snr2, f1, f2, ax, pstr, site, batch],
                                index=['dp_opt', 'wopt', 'evecs', 'evals', 'evec_sim', 'dU', 'dp_diag', 'dr_space', 'trials',
                                    'snr1', 'snr2', 'f1', 'f2', 'dr_weights', 'pair', 'site', 'batch']).T)
                    # FA trials (all targets)
                    dp, _, evals, evecs, evec_sim, dU = compute_dprime(fa.dot(ax.T).T, t.dot(ax.T).T, wopt=wopt)
                    dp_diag, _, _, _, _, _ = compute_dprime(fa.dot(ax.T).T, t.dot(ax.T).T, diag=True)
                    df = df.append(pd.DataFrame(data=[dp, wopt, evecs, evals, evec_sim, dU, dp_diag, category, 'fa',
                                snr1, snr2, f1, f2, ax, pstr, site, batch],
                                index=['dp_opt', 'wopt', 'evecs', 'evals', 'evec_sim', 'dU', 'dp_diag', 'dr_space', 'trials',
                                    'snr1', 'snr2', 'f1', 'f2', 'dr_weights', 'pair', 'site', 'batch']).T)


                    edges = np.append(np.quantile(range(0, c.shape[0]), [0.5]).astype(int), c.shape[0])
                    sedges = np.append(0, edges[:-1])
                    earc = c[sedges[0]:edges[0]]
                    latc = c[sedges[1]:edges[1]]
                    edges = np.append(np.quantile(range(0, t.shape[0]), [0.5]).astype(int), t.shape[0])
                    sedges = np.append(0, edges[:-1])
                    eart = t[sedges[0]:edges[0]]
                    latt = t[sedges[1]:edges[1]]
                    # for early trials
                    dp, _, evals, evecs, evec_sim, dU = compute_dprime(earc.dot(ax.T).T, eart.dot(ax.T).T, wopt=wopt)
                    dp_diag, _, _, _, _, _ = compute_dprime(earc.dot(ax.T).T, eart.dot(ax.T).T, diag=True)
                    df = df.append(pd.DataFrame(data=[dp, wopt, evecs, evals, evec_sim, dU, dp_diag, category, 'early',
                                snr1, snr2, f1, f2, ax, pstr, site, batch],
                                index=['dp_opt', 'wopt', 'evecs', 'evals', 'evec_sim', 'dU', 'dp_diag', 'dr_space', 'trials',
                                    'snr1', 'snr2', 'f1', 'f2', 'dr_weights', 'pair', 'site', 'batch']).T)
                    # for late trials
                    dp, _, evals, evecs, evec_sim, dU = compute_dprime(latc.dot(ax.T).T, latt.dot(ax.T).T, wopt=wopt)
                    dp_diag, _, _, _, _, _ = compute_dprime(latc.dot(ax.T).T, latt.dot(ax.T).T, diag=True)
                    df = df.append(pd.DataFrame(data=[dp, wopt, evecs, evals, evec_sim, dU, dp_diag, category, 'late',
                                snr1, snr2, f1, f2, ax, pstr, site, batch],
                                index=['dp_opt', 'wopt', 'evecs', 'evals', 'evec_sim', 'dU', 'dp_diag', 'dr_space', 'trials',
                                    'snr1', 'snr2', 'f1', 'f2', 'dr_weights', 'pair', 'site', 'batch']).T)

        # for each sound, measure variance in sliding window of X time bins
        # do this in PC space of targets/catch, dDR space, and just raw variance
        for ep in catch + targets:
            r = ct_resp[ep][:, :, start:end].mean(axis=-1)
            rpc = r.dot(pc_axes.T)
            rtdr = r.dot(tdr_axes.T)
            edges = np.append(np.quantile(range(0, r.shape[0]), [0.5]).astype(int), r.shape[0])
            sedges = np.append(0, edges[:-1])
            qt = [1, 2]
            for i, (s, e, q) in enumerate(zip(sedges, edges, qt)):
                inds = range(s, e)
                var_pca = np.var(rpc[inds, :], axis=0).sum()
                var_tdr = np.var(rtdr[inds, :], axis=0).sum()
                var_all = np.var(r[inds, :], axis=0).sum()
                snr = thelp.get_snrs([ep])[0]
                df2 = df2.append(pd.DataFrame(data=[var_all, var_pca, var_tdr, q, ep, snr, site, batch],
                                            index=['all', 'pca', 'tdr', 'qt', 'epoch', 'snr', 'site', 'batch']).T)
            # finally, project passive data

        
        #f.tight_layout()

        #f.canvas.set_window_title(site)

dtypes = {
    'dp_opt': 'float32',
    'wopt': 'object',
    'evecs': 'object',
    'evals': 'object',
    'evec_sim': 'object',
    'dU': 'object',
    'dp_diag': 'float32',
    'dr_space': 'object',
    'trials': 'object',
    'snr1': 'float32',
    'snr2': 'float32',
    'f1': 'float32',
    'f2': 'float32',
    'dr_weights': 'object',
    'pair': 'object',
    'site': 'object',
    'batch': 'float32',
}
dtypes2 = {
    'all': 'float32',
    'pca': 'float32',
    'tdr': 'float32',
    'qt': 'float32',
    'epoch': 'object',
    'snr': 'float32',
    'site': 'object',
    'batch': 'float32'
}
df = df.astype(dtypes)
df2 = df2.astype(dtypes2)


dU_mag = df['dU'].apply(lambda x: np.linalg.norm(x))
df['dU_mag'] = dU_mag

val = 'dU_mag'
f, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.plot(df[(df.trials=='fa') & (df.dr_space=='tdr_overall') & (df.batch==324)][val], 
            df[(df.trials=='cr') & (df.dr_space=='tdr_overall') & (df.batch==324)][val], 'o')

ax.plot(df[(df.trials=='fa') & (df.dr_space=='tdr_overall') & (df.batch==325)][val], 
            df[(df.trials=='cr') & (df.dr_space=='tdr_overall') & (df.batch==325)][val], 'o')
ma = np.max(ax.get_xlim() + ax.get_ylim())
ax.plot([0, ma], [0, ma], 'k--')
ax.set_xlabel('FA')
ax.set_ylabel('CR')

f.tight_layout()


# plot early vs. late variace
space = 'all'
early = df2[df2.qt==1]
late = df2[df2.qt==2]

ms = 5
f, ax = plt.subplots(1, 3, figsize=(6, 4), sharey=True)

# all data
lat = late[late.snr==-np.inf][space]
ear = early[late.snr==-np.inf][space]
ax[0].plot([[0]*len(ear), [1]*len(lat)], [ear, lat], color='lightgrey', zorder=0)
ax[0].errorbar([0, 1], [ear.mean(), lat.mean()], yerr=[ear.sem(), lat.sem()], 
                capsize=3, lw=2, color='k', marker='o', markersize=6)

pval = ss.wilcoxon(ear, lat).pvalue
ax[0].set_title(f"Catch \n p: {round(pval, 3)}")

ax[0].set_xticks([0, 1])
ax[0].set_xticklabels(['Early', 'Late'], rotation=45)
ax[0].set_xlim([-0.2, 1.2])
ax[0].set_ylabel('Total Variance')


# low SNR targets
lat = late[late.snr==-5][space]
ear = early[early.snr==-5][space]
ax[1].plot([[0]*len(ear), [1]*len(lat)], [ear, lat], color='lightgrey', zorder=0)
ax[1].errorbar([0, 1], [ear.mean(), lat.mean()], yerr=[ear.sem(), lat.sem()], 
                capsize=3, lw=2, color='k', marker='o', markersize=6)

pval = ss.wilcoxon(ear, lat).pvalue
ax[1].set_title(f"Low SNR \n p: {round(pval, 3)}")

ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(['Early', 'Late'], rotation=45)
ax[1].set_xlim([-0.2, 1.2])
ax[1].set_ylabel('Total Variance')


# high SNR targets
lat = late[(late.snr==0) | (late.snr==np.inf)][space]
ear = early[(early.snr==0) | (early.snr==np.inf)][space]
ax[2].plot([[0]*len(ear), [1]*len(lat)], [ear, lat], color='lightgrey', zorder=0)
ax[2].errorbar([0, 1], [ear.mean(), lat.mean()], yerr=[ear.sem(), lat.sem()], 
                capsize=3, lw=2, color='k', marker='o', markersize=6)

pval = ss.wilcoxon(ear, lat).pvalue
ax[2].set_title(f"High SNR \n p: {round(pval, 3)}")

ax[2].set_xticks([0, 1])
ax[2].set_xticklabels(['Early', 'Late'], rotation=45)
ax[2].set_xlim([-0.2, 1.2])
ax[2].set_ylabel('Total Variance')

f.tight_layout()

space = ['dp_opt']
early = df[df.trials=='early']
late = df[df.trials=='late']


# dprime
space = 'dp_diag'
early = df[(df.dr_space=='tdr_pair') & (df.trials=='early')]
late = df[(df.dr_space=='tdr_pair') & (df.trials=='late')]

ms = 5
f, ax = plt.subplots(1, 2, figsize=(4, 4), sharey=True)

# low SNR targets
lat = late[late.snr2==-5][space]
ear = early[early.snr2==-5][space]
ax[0].plot([[0]*len(ear), [1]*len(lat)], [ear, lat], color='lightgrey', zorder=0)
ax[0].errorbar([0, 1], [ear.mean(), lat.mean()], yerr=[ear.sem(), lat.sem()], 
                capsize=3, lw=2, color='k', marker='o', markersize=6)

pval = ss.wilcoxon(ear, lat).pvalue
ax[0].set_title(f"Low SNR \n p: {round(pval, 3)}")

ax[0].set_xticks([0, 1])
ax[0].set_xticklabels(['Early', 'Late'], rotation=45)
ax[0].set_xlim([-0.2, 1.2])
ax[0].set_ylabel(r"$d'$")


# high SNR targets
lat = late[(late.snr2==0) | (late.snr2==np.inf)][space]
ear = early[(early.snr2==0) | (early.snr2==np.inf)][space]
ax[1].plot([[0]*len(ear), [1]*len(lat)], [ear, lat], color='lightgrey', zorder=0)
ax[1].errorbar([0, 1], [ear.mean(), lat.mean()], yerr=[ear.sem(), lat.sem()], 
                capsize=3, lw=2, color='k', marker='o', markersize=6)

pval = ss.wilcoxon(ear, lat).pvalue
ax[1].set_title(f"High SNR \n p: {round(pval, 3)}")

ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(['Early', 'Late'], rotation=45)
ax[1].set_xlim([-0.2, 1.2])
ax[1].set_ylabel(r"$d'$")

f.tight_layout()

plt.show()