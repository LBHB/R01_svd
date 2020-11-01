"""
Two example neurons, with waveforms
"""

from nems_lbhb.baphy_experiment import BAPHYExperiment
import matplotlib.pyplot as plt
import numpy as np
import nems_lbhb.baphy_io as io
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 6

savefig = True
parmfile = '/auto/data/daq/Armillaria/ARM004/ARM004e18_p_NON.m'
#parmfile = '/auto/data/daq/Armillaria/ARM004/ARM004d05_p_NON.m'
animal = 'Armillaria'
unit1 = 'ARM004e-61-1'
unit2 = 'ARM004e-55-1'
rasterfs = 1000
recache = False
options = {'resp': True, 'rasterfs': rasterfs}

manager = BAPHYExperiment(parmfile=parmfile)
tstart = -0.02
tend = 0.1

rec = manager.get_recording(recache=recache, **options)
rec['resp'] = rec['resp'].rasterize()

prestim = rec['resp'].extract_epoch('PreStimSilence').shape[-1] / rasterfs
m = rec.copy().and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
poststim = (rec['resp'].extract_epoch('REFERENCE', mask=m['mask'], allow_incomplete=True).shape[-1] / rasterfs) + prestim
lim = (-prestim, tend ) 
lim = (tstart, tend)
s = 1

# get light on / off
opt_data = rec['resp'].epoch_to_signal('LIGHTON')
opt_mask = opt_data.extract_epoch('REFERENCE').mean(axis=(1,2)) > 0
opt_s_stop = (np.argwhere(np.diff(opt_data.extract_epoch('REFERENCE')[opt_mask, :, :][0].squeeze())) + 1) / rasterfs

# make figures
f, ax = plt.subplots(2, 2, figsize=(5, 4))

# ================= non-tagged unit ========================
r = rec['resp'].extract_channels([unit2]).extract_epoch('REFERENCE').squeeze()

# psth
on = r[opt_mask, :].mean(axis=0) * options['rasterfs']
on_sem = r[opt_mask, :].std(axis=0) / np.sqrt(opt_mask.sum()) * options['rasterfs']
t = np.arange(0, on.shape[-1] / options['rasterfs'], 1/options['rasterfs']) - prestim
ax[1, 0].plot(t, on, color='blue')
ax[1, 0].fill_between(t, on-on_sem, on+on_sem, alpha=0.3, lw=0, color='blue')
off = r[~opt_mask, :].mean(axis=0) * options['rasterfs']
off_sem = r[~opt_mask, :].std(axis=0) / np.sqrt((~opt_mask).sum()) * options['rasterfs']
t = np.arange(0, off.shape[-1] / options['rasterfs'], 1/options['rasterfs']) - prestim
ax[1, 0].plot(t, off, color='grey')
ax[1, 0].fill_between(t, off-off_sem, off+off_sem, alpha=0.3, lw=0, color='grey')
ax[1, 0].set_ylabel('Spk / sec')
ax[1, 0].set_xlim(lim[0], lim[1])

# spike raster / light onset/offset
st = np.where(r[opt_mask, :])
ax[0, 0].scatter((st[1] / rasterfs) - prestim, st[0], s=s, color='b')
offset = st[0].max()
st = np.where(r[~opt_mask, :])
ax[0, 0].scatter((st[1] / rasterfs) - prestim, st[0]+offset, s=s, color='grey')
for ss in opt_s_stop:
    ax[0, 0].axvline(ss - prestim, linestyle='--', color='lime')

ax[0, 0].set_title(unit2)
ax[0, 0].set_ylabel('Rep')
ax[0, 0].set_xlim(lim[0], lim[1])
# add inset for mwf
ax2 = plt.axes([ax[1, 0].colNum, ax[1, 0].colNum, ax[1, 0].rowNum, ax[1, 0].rowNum])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax[1, 0], [0.5,0.5,0.5,0.5])
ax2.set_axes_locator(ip)
mwf = io.get_mean_spike_waveform(unit2, animal)
ax2.plot(mwf, color='red')
ax2.axis('off') 

ax[1, 0].set_xlabel('Time from light onset (sec)')

# =================== tagged unit =========================
r = rec['resp'].extract_channels([unit1]).extract_epoch('REFERENCE').squeeze()

# psth
on = r[opt_mask, :].mean(axis=0) * options['rasterfs']
on_sem = r[opt_mask, :].std(axis=0) / np.sqrt(opt_mask.sum()) * options['rasterfs']
t = np.arange(0, on.shape[-1] / options['rasterfs'], 1/options['rasterfs']) - prestim
ax[1, 1].plot(t, on, color='blue')
ax[1, 1].fill_between(t, on-on_sem, on+on_sem, alpha=0.3, lw=0, color='blue')
off = r[~opt_mask, :].mean(axis=0) * options['rasterfs']
off_sem = r[~opt_mask, :].std(axis=0) / np.sqrt((~opt_mask).sum()) * options['rasterfs']
t = np.arange(0, off.shape[-1] / options['rasterfs'], 1/options['rasterfs']) - prestim
ax[1, 1].plot(t, off, color='grey')
ax[1, 1].fill_between(t, off-off_sem, off+off_sem, alpha=0.3, lw=0, color='grey')
ax[1, 1].set_ylabel('Spk / sec')
ax[1, 1].set_xlim(lim[0], lim[1])

# spike raster / light onset/offset
st = np.where(r[opt_mask, :])
ax[0, 1].scatter((st[1] / rasterfs) - prestim, st[0], s=s, color='b')
offset = st[0].max()
st = np.where(r[~opt_mask, :])
ax[0, 1].scatter((st[1] / rasterfs) - prestim, st[0]+offset, s=s, color='grey')
for ss in opt_s_stop:
    ax[0, 1].axvline(ss - prestim, linestyle='--', color='lime')

ax[0, 1].set_title(unit1)
ax[0, 1].set_ylabel('Rep')
ax[0, 1].set_xlim(lim[0], lim[1])
# add inset for mwf
ax2 = plt.axes([ax[1, 1].colNum, ax[1, 1].colNum, ax[1, 1].rowNum, ax[1, 1].rowNum])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax[1, 1], [0.5,0.5,0.5,0.5])
ax2.set_axes_locator(ip)
mwf = io.get_mean_spike_waveform(unit1, animal)
ax2.plot(mwf, color='red')
ax2.axis('off') 

ax[1, 1].set_xlabel('Time from light onset (sec)')

# share firing rate axes
m = np.max(ax[1, 0].get_ylim() + ax[1, 1].get_ylim())
ax[1, 0].set_ylim([0, m])
ax[1, 1].set_ylim([0, m])

f.tight_layout()

if savefig:
    f.savefig(DIR + '/results/figures/phototagging_example.pdf')

plt.show()