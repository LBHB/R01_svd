"""
Active vs. passive scatter plot for three categories. 
Summary of selective effects (invariance)
"""
import scipy.stats as ss
import statistics as stats
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
norm_delta = False
ylim = 5
figsave = DIR + 'results/figures/decodng_summary.pdf'
bootstrap_pvalue = True

df = pd.read_pickle(DIR + 'results/res.pickle')
df.index = df.pair

val = 'dp_opt'
df[val] = np.sqrt(df[val])
m = 8
ms = 4

batches = [324, 325]

tar_mask = (df.tar_tar) & (df.tdr_overall) & (~df.pca) & df.batch.isin(batches) & (df.f1 == df.f2) & ~df.sim1 & (df.trials=='all')
cat_mask = (df.cat_tar) & (df.tdr_overall) & (~df.pca) & df.batch.isin(batches) & (df.f1 == df.f2) & ~df.sim1 & (df.trials=='all')
ref_mask = (df.ref_ref) & (df.tdr_overall) & (~df.pca) & df.batch.isin(batches) & ~df.sim1 & (df.trials=='all')

f, ax = plt.subplots(2, 1, figsize=(2, 4), sharey=False)
ticks = [0, 0.25, 0.75, 1, 1.5, 1.75]
# normalize to mean passive across all categories
norm = pd.concat([df[ref_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val], 
            df[tar_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val], 
            df[cat_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val]]).mean()

# ==== A1 ===== 
for s in df[(df.area=='A1')].site.unique():
    p = df[ref_mask & ~df.active & (df.area=='A1') & (df.site==s)][val]
    a = df[ref_mask & df.active & (df.area=='A1') & (df.site==s)][val]
    ax[0].errorbar(ticks[0:2], [p.mean() / norm, a.mean() / norm], yerr=[p.sem(), a.sem()], capsize=3, 
                        color='mediumblue', markeredgecolor='k', marker='o', markersize=ms)

    p = df[tar_mask & ~df.active & (df.area=='A1') & (df.site==s)][val]
    a = df[tar_mask & df.active & (df.area=='A1') & (df.site==s)][val]
    ax[0].errorbar(ticks[2:4], [p.mean() / norm, a.mean() / norm], yerr=[p.sem(), a.sem()], capsize=3, 
                        color='coral', markeredgecolor='k', marker='o', markersize=ms)

    p = df[cat_mask & ~df.active & (df.area=='A1') & (df.site==s)][val]
    a = df[cat_mask & df.active & (df.area=='A1') & (df.site==s)][val]
    ax[0].errorbar(ticks[4:], [p.mean() / norm, a.mean() / norm], yerr=[p.sem(), a.sem()], capsize=3, 
                        color='lightgrey', markeredgecolor='k', marker='o', markersize=ms)


ax[0].axhline(1, linestyle='--', color='grey')
ax[0].set_ylabel(r"$d'$ normalized to mean passive")
ax[0].set_xticks(ticks)
ax[0].set_xticklabels(['Pas', 'Act', 'Pas', 'Act', 'Pas', 'Act'], rotation=45)

# normalize to mean passive across all categories
norm = pd.concat([df[ref_mask & ~df.active & (df.area=='PEG')].groupby(by='site').mean()[val], 
            df[tar_mask & ~df.active & (df.area=='PEG')].groupby(by='site').mean()[val], 
            df[cat_mask & ~df.active & (df.area=='PEG')].groupby(by='site').mean()[val]]).mean()
# ==== PEG ===== 
for s in df[(df.area=='PEG')].site.unique():
    p = df[ref_mask & ~df.active & (df.area=='PEG') & (df.site==s)][val]
    a = df[ref_mask & df.active & (df.area=='PEG') & (df.site==s)][val]
    ax[1].errorbar(ticks[0:2], [p.mean() / norm, a.mean() / norm], yerr=[p.sem(), a.sem()], capsize=3, 
                        color='mediumblue', markeredgecolor='k', marker='o', markersize=ms)

    p = df[tar_mask & ~df.active & (df.area=='PEG') & (df.site==s)][val]
    a = df[tar_mask & df.active & (df.area=='PEG') & (df.site==s)][val]
    ax[1].errorbar(ticks[2:4], [p.mean() / norm, a.mean() / norm], yerr=[p.sem(), a.sem()], capsize=3, 
                        color='coral', markeredgecolor='k', marker='o', markersize=ms)

    p = df[cat_mask & ~df.active & (df.area=='PEG') & (df.site==s)][val]
    a = df[cat_mask & df.active & (df.area=='PEG') & (df.site==s)][val]
    ax[1].errorbar(ticks[4:], [p.mean() / norm, a.mean() / norm], yerr=[p.sem(), a.sem()], capsize=3, 
                        color='lightgrey', markeredgecolor='k', marker='o', markersize=ms)
ax[1].axhline(1, linestyle='--', color='grey')
ax[1].set_ylabel(r"$d'$ normalized to mean passive")
ax[1].set_xticks(ticks)
ax[1].set_xticklabels(['Pas', 'Act', 'Pas', 'Act', 'Pas', 'Act'], rotation=45)

f.tight_layout()

# final ? summary plot
np.random.seed(123)
ticks = [0, 1, 2]
ms = 5
sd = 0.1
f, ax = plt.subplots(2, 2, figsize=(4, 4))


# ================================== ABS DPRIME SCATTER ==============================
ax[0, 0].scatter(df[ref_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val],
                    df[ref_mask & df.active & (df.area=='A1')].groupby(by='site').mean()[val], 
                    color='mediumblue', edgecolor='k', s=30, label='Ref vs. Ref')
ax[0, 0].scatter(df[tar_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val],
                    df[tar_mask & df.active & (df.area=='A1')].groupby(by='site').mean()[val], 
                    color='coral', edgecolor='k', s=30, label='Tar vs. Tar')
ax[0, 0].scatter(df[cat_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val],
                    df[cat_mask & df.active & (df.area=='A1')].groupby(by='site').mean()[val], 
                    color='lightgrey', edgecolor='k', s=30, label='Cat vs. Tar')

if bootstrap_pvalue:
    data = df[(ref_mask | tar_mask | cat_mask) & df.active & (df.area=='A1')][val] - df[(ref_mask | tar_mask | cat_mask) & ~df.active & (df.area=='A1')][val]
    data.index = df[(ref_mask | tar_mask | cat_mask) & df.active & (df.area=='A1')].site
    d = {s: data.loc[s].values.squeeze() for s in data.index.get_level_values(0).unique()}
    bs = stats.get_bootstrapped_sample(d, even_sample=False, nboot=1000)
    pvalue = stats.get_direct_prob(bs, np.zeros(len(bs)))[0]
else:
    data = pd.concat([df[ref_mask & df.active & (df.area=='A1')].groupby(by='site').mean()[val] - df[ref_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val],
                      df[tar_mask & df.active & (df.area=='A1')].groupby(by='site').mean()[val] - df[tar_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val],
                      df[cat_mask & df.active & (df.area=='A1')].groupby(by='site').mean()[val] - df[cat_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val]])
    pvalue = ss.wilcoxon(data.values).pvalue

ax[0, 0].legend(frameon=False)
ax[0, 0].set_xlabel('Passive')
ax[0, 0].set_ylabel('Active')
ax[0, 0].set_title('A1, pval: {:.2e}'.format(pvalue))


P = pd.concat([df[ref_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val],
                df[tar_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val],
                df[cat_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val]])
A = pd.concat([df[ref_mask & df.active & (df.area=='A1')].groupby(by='site').mean()[val],
                df[tar_mask & df.active & (df.area=='A1')].groupby(by='site').mean()[val],
                df[cat_mask & df.active & (df.area=='A1')].groupby(by='site').mean()[val]])
pval = ss.wilcoxon(P, A).pvalue
print(f"Active vs. passive dprime for all sites/categories in A1, pval: {pval}")

ax[1, 0].scatter(df[ref_mask & ~df.active & (df.area=='PEG')].groupby(by='site').mean()[val],
                    df[ref_mask & df.active & (df.area=='PEG')].groupby(by='site').mean()[val], 
                    color='mediumblue', edgecolor='k', s=30, label='Ref vs. Ref')
ax[1, 0].scatter(df[tar_mask & ~df.active & (df.area=='PEG')].groupby(by='site').mean()[val],
                    df[tar_mask & df.active & (df.area=='PEG')].groupby(by='site').mean()[val], 
                    color='coral', edgecolor='k', s=30, label='Tar vs. Tar')
ax[1, 0].scatter(df[cat_mask & ~df.active & (df.area=='PEG')].groupby(by='site').mean()[val],
                    df[cat_mask & df.active & (df.area=='PEG')].groupby(by='site').mean()[val], 
                    color='lightgrey', edgecolor='k', s=30, label='Cat vs. Tar')

if bootstrap_pvalue:
    data = df[(ref_mask | tar_mask | cat_mask) & df.active & (df.area=='PEG')][val] - df[(ref_mask | tar_mask | cat_mask) & ~df.active & (df.area=='PEG')][val]
    data.index = df[(ref_mask | tar_mask | cat_mask) & df.active & (df.area=='PEG')].site
    d = {s: data.loc[s].values.squeeze() for s in data.index.get_level_values(0).unique()}
    bs = stats.get_bootstrapped_sample(d, even_sample=False, nboot=1000)
    pvalue = stats.get_direct_prob(bs, np.zeros(len(bs)))[0]

else:
    data = pd.concat([df[ref_mask & df.active & (df.area=='PEG')].groupby(by='site').mean()[val] - df[ref_mask & ~df.active & (df.area=='PEG')].groupby(by='site').mean()[val],
                      df[tar_mask & df.active & (df.area=='PEG')].groupby(by='site').mean()[val] - df[tar_mask & ~df.active & (df.area=='PEG')].groupby(by='site').mean()[val],
                      df[cat_mask & df.active & (df.area=='PEG')].groupby(by='site').mean()[val] - df[cat_mask & ~df.active & (df.area=='PEG')].groupby(by='site').mean()[val]])
    pvalue = ss.wilcoxon(data.values).pvalue
ax[1, 0].set_xlabel('Passive')
ax[1, 0].set_ylabel('Active')
ax[1, 0].set_title('PEG, pval: {:.2e}'.format(pvalue))

P = pd.concat([df[ref_mask & ~df.active & (df.area=='PEG')].groupby(by='site').mean()[val],
                df[tar_mask & ~df.active & (df.area=='PEG')].groupby(by='site').mean()[val],
                df[cat_mask & ~df.active & (df.area=='PEG')].groupby(by='site').mean()[val]])
A = pd.concat([df[ref_mask & df.active & (df.area=='PEG')].groupby(by='site').mean()[val],
                df[tar_mask & df.active & (df.area=='PEG')].groupby(by='site').mean()[val],
                df[cat_mask & df.active & (df.area=='PEG')].groupby(by='site').mean()[val]])
pval = ss.wilcoxon(P, A).pvalue
print(f"Active vs. passive dprime for all sites/categories in PEG, pval: {pval}")


m = np.max(ax[0, 0].get_xlim() + ax[0, 0].get_ylim() + ax[1, 0].get_xlim() + ax[1, 0].get_ylim())
ax[0, 0].plot([0, m], [0, m], linestyle='--', color='grey')
ax[1, 0].plot([0, m], [0, m], linestyle='--', color='grey')

# ================================== DELTA DPRIME ====================================

# ref - ref
rr_act = df[ref_mask & df.active & (df.area=='A1')][[val, 'site']].set_index('site')
rr_pass = df[ref_mask & ~df.active & (df.area=='A1')][[val, 'site']].set_index('site')
if norm_delta:
    rr_delt = (rr_act - rr_pass) / (rr_act + rr_pass)
else:
    rr_delt = rr_act - rr_pass
if bootstrap_pvalue:
    d = {s: rr_delt.loc[s].values.squeeze() for s in rr_delt.index.get_level_values(0).unique()}
    bs = stats.get_bootstrapped_sample(d, even_sample=False, nboot=1000)
    pvalue1 = stats.get_direct_prob(bs, np.zeros(len(bs)))[0]
else:
    pvalue1 = ss.wilcoxon(rr_delt.values.squeeze()).pvalue
ax[0, 1].scatter(np.random.normal(ticks[0], sd, len(rr_delt)),
                rr_delt, s=ms, alpha=0.1, color='mediumblue', edgecolor='none')
ax[0, 1].errorbar(ticks[0], rr_delt.groupby(level=0).mean().mean(), yerr=rr_delt.groupby(level=0).mean().sem(), capsize=3, 
                        color='mediumblue', markeredgecolor='k', marker='o', markersize=ms, ecolor='k')

# cat - tar
ct_act = df[cat_mask & df.active & (df.area=='A1')][[val, 'site']].set_index('site')
ct_pass = df[cat_mask & ~df.active & (df.area=='A1')][[val, 'site']].set_index('site')
if norm_delta:
    ct_delt = (ct_act - ct_pass) / (ct_act + ct_pass)
else:
    ct_delt = ct_act - ct_pass
if bootstrap_pvalue:
    d = {s: ct_delt.loc[s].values.squeeze() for s in ct_delt.index.get_level_values(0).unique()}
    bs = stats.get_bootstrapped_sample(d, even_sample=False, nboot=1000)
    pvalue2 = stats.get_direct_prob(bs, np.zeros(len(bs)))[0]
else:
    pvalue2 = ss.wilcoxon(ct_delt.values.squeeze()).pvalue
ax[0, 1].scatter(np.random.normal(ticks[1], sd, len(ct_delt)),
                ct_delt, s=ms, alpha=0.7, color='lightgrey', edgecolor='none')
ax[0, 1].errorbar(ticks[1], ct_delt.groupby(level=0).mean().mean(), yerr=ct_delt.groupby(level=0).mean().sem(), capsize=3, 
                        color='lightgrey', markeredgecolor='k', marker='o', markersize=ms, ecolor='k')

# tar - tar
tt_act = df[tar_mask & df.active & (df.area=='A1')][[val, 'site']].set_index('site')
tt_pass = df[tar_mask & ~df.active & (df.area=='A1')][[val, 'site']].set_index('site')
if norm_delta:
    tt_delt = (tt_act - tt_pass) / (tt_act + tt_pass)
else:
    tt_delt = tt_act - tt_pass
if bootstrap_pvalue:
    d = {s: (tt_delt.loc[s].values.squeeze() if tt_delt.loc[s].shape[0]>1 else tt_delt.loc[s].values) for s in tt_delt.index.get_level_values(0).unique()}
    bs = stats.get_bootstrapped_sample(d, even_sample=False, nboot=1000)
    pvalue3 = stats.get_direct_prob(bs, np.zeros(len(bs)))[0]
else:
    pvalue3 = ss.wilcoxon(tt_delt.values.squeeze()).pvalue
ax[0, 1].scatter(np.random.normal(ticks[2], sd, len(tt_delt)),
                tt_delt, s=ms, alpha=0.7, color='coral', edgecolor='none')
ax[0, 1].errorbar(ticks[2], tt_delt.groupby(level=0).mean().mean(), yerr=tt_delt.groupby(level=0).mean().sem(), capsize=3, 
                        color='coral', markeredgecolor='k', marker='o', markersize=ms, ecolor='k')

ax[0, 1].axhline(0, linestyle='--', color='grey', zorder=0)
ax[0, 1].set_ylabel(r"$\Delta d'$")
ax[0, 1].set_ylim((None, ylim))

ax[0, 1].set_title("p={:.2e}, p={:.2e}, p={:.2e}".format(pvalue1, pvalue2, pvalue3))

# ref - ref
rr_act = df[ref_mask & df.active & (df.area=='PEG')][[val, 'site']].set_index('site')
rr_pass = df[ref_mask & ~df.active & (df.area=='PEG')][[val, 'site']].set_index('site')
if norm_delta:
    rr_delt = (rr_act - rr_pass) / (rr_act + rr_pass)
else:
    rr_delt = rr_act - rr_pass
if bootstrap_pvalue:
    d = {s: rr_delt.loc[s].values.squeeze() for s in rr_delt.index.get_level_values(0).unique()}
    bs = stats.get_bootstrapped_sample(d, even_sample=False, nboot=1000)
    pvalue1 = stats.get_direct_prob(bs, np.zeros(len(bs)))[0]
else:
    pvalue1 = ss.wilcoxon(rr_delt.values.squeeze()).pvalue
ax[1, 1].scatter(np.random.normal(ticks[0], sd, len(rr_delt)),
                rr_delt, s=ms, alpha=0.1, color='mediumblue', edgecolor='none')
ax[1, 1].errorbar(ticks[0], rr_delt.groupby(level=0).mean().mean(), yerr=rr_delt.groupby(level=0).mean().sem(), capsize=3, 
                        color='mediumblue', markeredgecolor='k', marker='o', markersize=ms, ecolor='k')
# cat - tar
ct_act = df[cat_mask & df.active & (df.area=='PEG')][[val, 'site']].set_index('site')
ct_pass = df[cat_mask & ~df.active & (df.area=='PEG')][[val, 'site']].set_index('site')
if norm_delta:
    ct_delt = (ct_act - ct_pass) / (ct_act + ct_pass)
else:
    ct_delt = ct_act - ct_pass
if bootstrap_pvalue:
    d = {s: (ct_delt.loc[s].values.squeeze() if ct_delt.loc[s].shape[0]>1 else ct_delt.loc[s].values) for s in ct_delt.index.get_level_values(0).unique()}
    bs = stats.get_bootstrapped_sample(d, even_sample=False, nboot=1000)
    pvalue2 = stats.get_direct_prob(bs, np.zeros(len(bs)))[0]
else:
    pvalue2 = ss.wilcoxon(ct_delt.values.squeeze()).pvalue
ax[1, 1].scatter(np.random.normal(ticks[1], sd, len(ct_delt)),
                ct_delt, s=ms, alpha=0.7, color='lightgrey', edgecolor='none')
ax[1, 1].errorbar(ticks[1], ct_delt.groupby(level=0).mean().mean(), yerr=ct_delt.groupby(level=0).mean().sem(), capsize=3, 
                        color='lightgrey', markeredgecolor='k', marker='o', markersize=ms, ecolor='k')

# tar - tar
tt_act = df[tar_mask & df.active & (df.area=='PEG')][[val, 'site']].set_index('site')
tt_pass = df[tar_mask & ~df.active & (df.area=='PEG')][[val, 'site']].set_index('site')
if norm_delta:
    tt_delt = (tt_act - tt_pass) / (tt_act + tt_pass)
else:
    tt_delt = tt_act - tt_pass
if bootstrap_pvalue:
    d = {s: (tt_delt.loc[s].values.squeeze() if tt_delt.loc[s].shape[0]>1 else tt_delt.loc[s].values) for s in tt_delt.index.get_level_values(0).unique()}
    bs = stats.get_bootstrapped_sample(d, even_sample=False, nboot=1000)
    pvalue3 = stats.get_direct_prob(bs, np.zeros(len(bs)))[0]
else:
    pvalue3 = ss.wilcoxon(tt_delt.values.squeeze()).pvalue
ax[1, 1].scatter(np.random.normal(ticks[2], sd, len(tt_delt)),
                tt_delt, s=ms, alpha=0.7, color='coral', edgecolor='none')
ax[1, 1].errorbar(ticks[2], tt_delt.groupby(level=0).mean().mean(), yerr=tt_delt.groupby(level=0).mean().sem(), capsize=3, 
                        color='coral', markeredgecolor='k', marker='o', markersize=ms, ecolor='k')

ax[1, 1].axhline(0, linestyle='--', color='grey', zorder=0)
ax[1, 1].set_ylabel(r"$\Delta d'$")
ax[1, 1].set_ylim((None, ylim))

ax[1, 1].set_title("p={:.2e}, p={:.2e}, p={:.2e}".format(pvalue1, pvalue2, pvalue3))

f.tight_layout()

if savefig:
    f.savefig(figsave)

plt.show()
