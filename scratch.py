from nems_lbhb.baphy_experiment import BAPHYExperiment
from itertools import combinations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

np.random.seed(123)

batch = 324
site = 'CRD018d'
options = {'rasterfs': 10, 'resp': True, 'pupil': True}

manager = BAPHYExperiment(batch=batch, siteid=site[:7])
rec = manager.get_recording(**options)
rec['resp'] = rec['resp'].rasterize()

start = int(0.1 * options['rasterfs'])
end = int(0.3 * options['rasterfs'])

# extract a single target for active / passive
ep = 'TAR_3249+0dB+Noise'

rec_a = rec.and_mask(['HIT_TRIAL', 'MISS_TRIAL', 'CORRECT_REJECT_TRIAL'])
rec_p = rec.and_mask(['PASSIVE_EXPERIMENT'])

ra = rec['resp'].extract_epoch(ep, mask=rec_a['mask'])[:, :, start:end].mean(axis=-1)
rp = rec['resp'].extract_epoch(ep, mask=rec_p['mask'])[:, :, start:end].mean(axis=-1)

nNeurons = ra.shape[-1]

# subsample as many time as possible, cap at 100
res = pd.DataFrame(index=range(nNeurons), columns=['a_var', 'a_sem', 'p_var', 'p_sem'])
for i in range(1, nNeurons):
    samp_idx = [np.random.choice(range(0, ra.shape[-1]), i, replace=True) for j in range(ra.shape[-1])]
    a_var = []
    p_var = []
    for sidx in samp_idx:
        a = ra[:, sidx]
        p = rp[:, sidx]

        a_var.append(a.var(axis=0).sum())
        p_var.append(p.var(axis=0).sum())

    res.at[i, 'a_var'] = np.mean(a_var)
    res.at[i, 'a_sem'] = np.std(a_var) / np.sqrt(ra.shape[-1])
    res.at[i, 'p_var'] = np.mean(p_var)
    res.at[i, 'p_sem'] = np.std(p_var) / np.sqrt(ra.shape[-1])

dtypes = {'a_var': 'float32', 'p_var': 'float32',
            'a_sem': 'float32', 'p_sem': 'float32'}
res = res.astype(dtypes)

f, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.plot(range(nNeurons), res['a_var'], color='r', label='active')
ax.fill_between(np.arange(0, nNeurons), res['a_var']-res['a_sem'], 
        res['a_var']+res['a_sem'], color='red', alpha=0.2, lw=0)

ax.plot(range(nNeurons), res['p_var'], color='blue', label='passive')
ax.fill_between(np.arange(0, nNeurons), res['p_var']-res['p_sem'], 
        res['p_var']+res['p_sem'], color='blue', alpha=0.2, lw=0)


ax.plot(res['p_var'] - res['a_var'], color='k', linestyle='--', label='difference')

ax.legend()
ax.set_ylabel('Total variance')
ax.set_xlabel('Number of Neurons')

f.tight_layout()

plt.show()

    

