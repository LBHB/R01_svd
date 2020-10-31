"""
Generic summary of behavior performance
RT histogram over all data (mean of all training + recording sessions)
Psychometric function of DI (calculated as above ^^)
"""
import nems_lbhb.tin_helpers as thelp

import statistics
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.behavior import get_reaction_times
from nems_lbhb.behavior_plots import plot_RT_histogram
import nems.db as nd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
import datetime as dt

# recording load options
options = {"resp": False, "pupil": False, "rasterfs": 20}

runclass = 'TBP'
ed = '2020-08-05'
ld = '2020-08-07' #'2020-09-24'
ed = dt.datetime.strptime(ed, '%Y-%m-%d')
ld = dt.datetime.strptime(ld, '%Y-%m-%d')

# get list of parmfiles and sort by date (all files from one day go into one analysis??)
sql = "SELECT gDataRaw.resppath, gDataRaw.parmfile, pendate FROM gPenetration INNER JOIN gCellMaster ON (gCellMaster.penid = gPenetration.id)"\
                " INNER JOIN gDataRaw ON (gCellMaster.id = gDataRaw.masterid) WHERE" \
                " gDataRaw.runclass='{0}' and gDataRaw.bad=0 and gDataRaw.trials>50 and"\
                " gPenetration.animal = 'Cordyceps' and gDataRaw.behavior='active'".format(runclass)
d = nd.pd_query(sql)
d['date'] = d['pendate'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))

# screen for date
d = d[(d['date'] >= ed) & (d['date'] <= ld)]
d = d.sort_values(by='date')

# join path
d['parmfile_path'] = [os.path.join(d['resppath'].iloc[i], d['parmfile'].iloc[i]) for i in range(d.shape[0])]

# define set of unique dates
uDate = d['date'].unique()

# store RTs according to SNR
snr_strs = ['-inf', '-10', '-5', '0', 'inf']
rts = {k: [] for k in snr_strs}
for idx, ud in enumerate(uDate):
    print(f"Loading data from {ud}")
    parmfiles = d[d.date==ud].parmfile_path.values.tolist()
    # add catch to make sure "siteid" the same for all files
    sid = [p.split(os.path.sep)[-1][:7] for p in parmfiles]
    if np.any(np.array(sid) != sid[0]):
        bad_idx = (np.array(sid)!=sid[0])
        parmfiles = np.array(parmfiles)[~bad_idx].tolist()
    manager = BAPHYExperiment(parmfiles)

    # make sure only loaded actives
    pf_mask = [True if k['BehaveObjectClass']=='RewardTargetLBHB' else False for k in manager.get_baphy_exptparams()]
    if sum(pf_mask) == len(manager.parmfile):
        pass
    else:
        parmfiles = np.array(manager.parmfile)[pf_mask].tolist()
        manager = BAPHYExperiment(parmfiles)

    # get reaction times of targets, only for "valid" trials
    bev = manager.get_behavior_events(**options)
    bev = manager._stack_events(bev)
    bev = bev[bev.invalidTrial==False]
    _rts = get_reaction_times(manager.get_baphy_exptparams()[0], bev, **options)

    targets = _rts['Target'].keys()
    snrs = thelp.get_snrs(targets)
    for s, t in zip(snrs, targets):
        rts[str(s)].extend(_rts['Target'][t])

# get colormap for targets (kludgy, bc this is for many recordings + refs don't matter here)
targets = ['TAR_1000+'+snr+'+Noise' for snr in rts.keys()]
reference = ['STIM_1000']
BwG, gR = thelp.make_tbp_colormaps(reference, targets, use_tar_freq_idx=0)


f, ax = plt.subplots(1, 1, figsize=(6, 4), sharey=True)

bins = np.arange(0, 1.2, 0.001)
plot_RT_histogram(rts, bins=bins, ax=ax, cmap=gR, lw=2)

f.tight_layout()

plt.show()