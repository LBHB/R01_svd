"""
Generic summary of behavior performance
RT histogram over all data (mean of all training + recording sessions)
Psychometric function of DI (calculated as above ^^)
"""
import nems_lbhb.tin_helpers as thelp
from settings import DIR
import pickle
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
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 6
import datetime as dt

savefig = True
recache = True
# recording load options
options = {"resp": False, "pupil": False, "rasterfs": 20}

runclass = 'TBP'
ed = '2020-08-05'
ld = '2020-09-24'
ed = dt.datetime.strptime(ed, '%Y-%m-%d')
ld = dt.datetime.strptime(ld, '%Y-%m-%d')
min_trials = 50

# get list of parmfiles and sort by date (all files from one day go into one analysis??)
sql = "SELECT gDataRaw.resppath, gDataRaw.parmfile, pendate FROM gPenetration INNER JOIN gCellMaster ON (gCellMaster.penid = gPenetration.id)"\
                " INNER JOIN gDataRaw ON (gCellMaster.id = gDataRaw.masterid) WHERE" \
                " gDataRaw.runclass='{0}' and gDataRaw.bad=0 and gDataRaw.trials>{1} and"\
                " gPenetration.animal = 'Cordyceps' and gDataRaw.behavior='active'".format(runclass, min_trials)
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
if recache:
    snr_strs = ['-inf', '-10', '-5', '0', 'inf']
    rts = {k: [] for k in snr_strs}
    DI = {k: [] for k in snr_strs if ('-inf' not in k)}
    nSessions = {k: 0 for k in snr_strs}
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

        # get behavior performance
        performance = manager.get_behavior_performance(**options)

        # get reaction times of targets, only for "valid" trials
        bev = manager.get_behavior_events(**options)
        bev = manager._stack_events(bev)
        bev = bev[bev.invalidTrial==False]
        _rts = get_reaction_times(manager.get_baphy_exptparams()[0], bev, **options)

        targets = _rts['Target'].keys()
        cat = [t for t in targets if '-Inf' in t][0]
        snrs = thelp.get_snrs(targets)
        # keep only the freqs with same CF as catch
        #freqs = thelp.get_freqs(targets)
        #idx = [True if freq==thelp.get_freqs([cat])[0] else False for freq in freqs]
        #targets = np.array(list(targets))[idx].tolist()
        #snrs = [s for s, i in zip(snrs, idx) if i==True]
        for s, t in zip(snrs, targets):
            rts[str(s)].extend(_rts['Target'][t])
            nSessions[str(s)] += 1
            _t = t.split(':')[0]
            if '-Inf' not in _t:
                try:
                    DI[str(s)].extend([performance['LI'][_t+'_'+cat.split(':')[0]]])
                except:
                    DI[str(s)].extend([performance['LI'][_t+'+reminder_'+cat.split(':')[0]]])

    # cache results
    pickle.dump(rts, open(DIR+"/results/Cordyceps_rts.pickle", "wb"))
    pickle.dump(DI, open(DIR+"/results/Cordyceps_DI.pickle", "wb"))
    pickle.dump(nSessions, open(DIR+"/results/Cordyceps_nSessions.pickle", "wb"))

else:
    rts = pickle.load(open(DIR+"/results/Cordyceps_rts.pickle", "rb"))
    DI = pickle.load(open(DIR+"/results/Cordyceps_DI.pickle", "rb"))
    nSessions = pickle.load(open(DIR+"/results/Cordyceps_nSessions.pickle", "rb"))

# get colormap for targets (kludgy, bc this is for many recordings + refs don't matter here)
targets = ['TAR_1000+'+snr+'+Noise' for snr in rts.keys()]
reference = ['STIM_1000']
BwG, gR = thelp.make_tbp_colormaps(reference, targets, use_tar_freq_idx=0)

# don't double count inf -- (reminder / target should get lumped together for this)
nSessions['inf'] = np.max([v for k, v in nSessions.items() if k != 'inf'])

# nothing should have higher count than number of unique date (that just means there were multiple tars at that SNR)
mi = len(uDate)
nSessions = {k: (v if v <= mi else mi) for k, v in nSessions.items()}

legend = [s+ f' dB, n = {n}' if '-inf' not in s else f'Catch, n = {n}' for s, n in zip(rts.keys(), nSessions.values())]

f, ax = plt.subplots(1, 1, figsize=(2, 2), sharey=True)

bins = np.arange(0, 1.2, 0.001)
plot_RT_histogram(rts, bins=bins, ax=ax, cmap=gR, lw=2, legend=legend)

f.tight_layout()

if savefig:
    f.savefig(DIR + '/results/figures/Cordyceps_RT.pdf')

plt.show()