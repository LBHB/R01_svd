"""
possible cellds:
batch=325
cellid="CRD010b-03-1"
cellid="CRD011c-14-1"
cellid="CRD012b-01-1"
cellid="CRD013b-10-1"
batch=324
cellid="CRD016c-27-2"
cellid="CRD017c-01-1"
cellid="CRD018d-04-1"
cellid="CRD019b-10-1"
cellid="CRD009b-06-1"
"""

from sklearn.decomposition import PCA
import nems.db as nd
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

params = {'legend.fontsize': 12,
          'axes.labelsize': 12,
          'axes.titlesize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
mpl.rcParams.update(params)

from scipy.stats import linregress
from sklearn.decomposition import PCA

from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems.preprocessing import make_state_signal
from nems_lbhb.tin_helpers import sort_targets, compute_ellipse, load_tbp_recording, pb_regress, \
   get_sound_labels, plot_average_psths, site_tuning_avg, site_tuning_curves
from nems.xform_helper import fit_model_xform, load_model_xform
import nems.db as nd

#outpath="/auto/users/svd/docs/current/grant/r01_AC_pop_A1/eps"
outpath="../results/figures/lv_simulations"
recpath="/auto/users/svd/projects/pop_models/tbp"
savefigs=True

options = {'resp': True, 'pupil': True, 'rasterfs': 10}

## Load example

modelname= 'll.fs10.pup-ld-norm-st.pup.afl-reftar-aev_wc.Nx30-fir.1x6x30-relu.30-wc.30xR-lvl.R-dexp.R-sdexp2.SxR_tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont'

if len(sys.argv)>1:
    siteid=sys.argv[1]
else:
    siteid="CRD010b"

d = nd.pd_query(f"SELECT batch,cellid FROM Batches WHERE cellid like %s", (siteid+"%",))
batch=d['batch'].values[0]
cellid=d['cellid'].values[0]

print(f"cell {cellid} batch {batch}")

xf,ctx = load_model_xform(cellid, batch, modelname)

#cellid="CRD010b"
#xf,ctx = fit_model_xform(cellid, batch, modelname, returnModel=True)

rec=ctx['val']
modelspec=ctx['modelspec']

onsetsec = 0.1
offsetsec = 0.3

onset = int(onsetsec * options['rasterfs'])
offset = int(offsetsec * options['rasterfs'])

ref_stim, tar_stim, all_stim = get_sound_labels(rec)


## Fit LV weights to reproduce corr stats

from scipy.optimize import fmin, minimize
from nems.preprocessing import generate_psth_from_resp
rm=generate_psth_from_resp(rec)
rm=rm.apply_mask()

psth = rm['psth_sp']._data
resp = rm['resp']._data
state = rm['state']._data.copy()
pred = rm['pred']._data

cellcount=resp.shape[0]
state_count=state.shape[0]

cmap='bwr'

# make lv_count=1 if you want to have a single, non-modulated lv
# lv_count=1  # static
# lv_count = 2   # pupil only
lv_count=state.shape[0]

indep_noise = np.random.randn(*resp.shape)
lv = np.random.randn(*state[:lv_count,:].shape)
pred0 = pred.copy()

actual_cc = np.cov(resp-pred0)
u,s,vh=np.linalg.svd(actual_cc)
pc1=u[:,0]

# compute noise correlation for active and passive conditions separately
aidx = np.sum(state[2:,:],axis=0)>0
pidx = np.sum(state[2:,:],axis=0)==0
active_cc = np.cov(resp[:,aidx]-pred0[:,aidx])
passive_cc = np.cov(resp[:,pidx]-pred0[:,pidx])

# specialized function to apply weighted LV to each neurons' prediction
# pred = pred_0 + (d+g*state) * lv  , with d,g as gain and offset
def lv_mod(d, g, state, lv, pred, showdetails=False):
    pred=pred.copy()
    if showdetails:
        f,ax=plt.subplots(d.shape[1],1,figsize=(10,5))
        if d.shape[1]==1:
            ax=[ax]
    for l in range(d.shape[1]):
        pred += (d[:,l:(l+1)] + g[:,l:(l+1)]*state[l:(l+1),:]) * lv[l:(l+1),:]
        if showdetails:
            ax[l].imshow((d[:,l:(l+1)] + g[:,l:(l+1)]*state[l:(l+1),:]), aspect='auto', interpolation='none', origin='lower', cmap=cmap)
    return pred 

# specialized cost function to compute error between predicted and actual noise correlation matrices
def cc_err(w, pred, indep_noise, lv, pred0, actual_cc):
    _w=np.reshape(w,[-1, lv_count*3])
    p = lv_mod(_w[:,lv_count:(lv_count*2)], _w[:,(lv_count*2):(lv_count*3)], state, lv, pred) + (_w[:,0:lv_count] @ state) *indep_noise
    pascc = np.cov(p[:,pidx] - pred0[:,pidx])
    actcc = np.cov(p[:,aidx] - pred0[:,aidx])
    E = np.sum((pascc-passive_cc)**2) / np.sum(passive_cc**2) + np.sum((actcc-active_cc)**2) / np.sum(active_cc**2)
    return E

## fit the LV weights

# initialize
w0 = np.zeros((cellcount,lv_count*3))
w0[:,0]=0.05
w0[:,lv_count*2]=pc1/10

# first fit without independent noise to push out to LVs
print('fit, round 1...')
res = minimize(cc_err, w0, args=(pred, indep_noise*0, lv, pred0, actual_cc), method='L-BFGS-B')
w1=np.reshape(res.x,[-1, lv_count*3])

# second fit WITH independent noise to allow for independent noise
print('fit, round 2...')
res = minimize(cc_err, w1, args=(pred, indep_noise, lv, pred0, actual_cc), method='L-BFGS-B')
w=np.reshape(res.x,[-1, lv_count*3])

## generate predictions with indep noise and LV
pred_data = rec['pred']._data.copy()
mm = rec['mask']._data[0,:]

pred_indep = pred + (w[:,0:lv_count] @ state) *indep_noise
pred_data[:,mm] = pred_indep
rec.signals['pred_indep'] = rec['pred']._modified_copy(data=pred_data)

pred_lv = lv_mod(w[:,lv_count:(lv_count*2)], w[:,(lv_count*2):(lv_count*3)], state, lv, pred, showdetails=True) + (w[:,0:lv_count] @ state) *indep_noise
pred_data = rec['pred']._data.copy()
pred_data[:,mm] = pred_lv
rec.signals['pred_lv'] = rec['pred']._modified_copy(data=pred_data)


## display noise corr. matrices
f,ax = plt.subplots(2,4, figsize=(8,4), sharex=True, sharey=True)

pred0=pred.copy()

aidx = np.sum(state[2:,:],axis=0)>0
pidx = np.sum(state[2:,:],axis=0)==0
active_cc = np.cov(resp[:,aidx]-pred0[:,aidx])
passive_cc = np.cov(resp[:,pidx]-pred0[:,pidx])

mm=np.max(np.abs(passive_cc)) * 0.6

pas_cc = np.cov(pred[:,pidx]-pred0[:,pidx])
act_cc = np.cov(pred[:,aidx]-pred0[:,aidx])
ax[0,0].imshow(pas_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr')
ax[1,0].imshow(act_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr')
ax[0,0].set_title(siteid + ' pred')

ax[0,0].set_ylabel('passive')
ax[1,0].set_ylabel('active')

pas_cc = np.cov(pred_indep[:,pidx]-pred0[:,pidx])
act_cc = np.cov(pred_indep[:,aidx]-pred0[:,aidx])
ax[0,1].imshow(pas_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr')
ax[1,1].imshow(act_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr')
ax[0,1].set_title('pred + indep noise')

pas_cc = np.cov(pred_lv[:,pidx]-pred0[:,pidx])
act_cc = np.cov(pred_lv[:,aidx]-pred0[:,aidx])
ax[0,2].imshow(pas_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr')
ax[1,2].imshow(act_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr')
ax[0,2].set_title('pred + indep + lv')

ax[0,3].imshow(passive_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr')
ax[1,3].imshow(active_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr')
ax[0,3].set_title('actual resp')

if savefigs:
    outfile = f"noise_corr_sim_{siteid}_{batch}.pdf"
    print(f"saving to {outpath}/{outfile}")
    f.savefig(f"{outpath}/{outfile}")


## find PCA and TDR axes

from nems_lbhb.dimensionality_reduction import TDR

# can't simply extract evoked for refs because can be longer/shorted if it came after target 
# and / or if it was the last stim.So, masking prestim / postim doesn't work.Do it manually
d1 = rec['resp'].extract_epoch('REFERENCE', mask=rec['mask'])
d2 = rec['resp'].extract_epoch('TARGET', mask=rec['mask'])

d1=np.mean(d1[:,:,2:4], axis=2)
d2=np.mean(d2[:,:,2:4], axis=2)

tdr = TDR()
tdr.fit(d1,d2)
tdr_axes = tdr.weights

# can't simply extract evoked for refs because can be longer/shorted if it came after target 
# and / or if it was the last stim.So, masking prestim / postim doesn't work.Do it manually
d = rec['resp'].extract_epochs(all_stim, mask=rec['mask'])
d = {k: v[~np.isnan(v[:, :, onset:offset].sum(axis=(1, 2))), :, :] for (k, v) in d.items()}
d = {k: v[:, :, onset:offset] for (k, v) in d.items()}

Rall_u = np.vstack([d[k].sum(axis=2).mean(axis=0) for k in d.keys()])

pca = PCA(n_components=2)
pca.fit(Rall_u)
pc_axes = pca.components_


## project onto one or the other
#a=tdr_axes
a=pc_axes

rt=rec.copy()

# project onto first two PCs
rt['rpc'] = rt['resp']._modified_copy(rt['resp']._data.T.dot(a.T).T[0:2, :])
rt['ppc_pred'] = rt['pred']._modified_copy(rt['pred']._data.T.dot(a.T).T[0:2, :])
rt['ppc_indep'] = rt['pred_indep']._modified_copy(rt['pred_indep']._data.T.dot(a.T).T[0:2, :])
rt['ppc_lv'] = rt['pred_lv']._modified_copy(rt['pred_lv']._data.T.dot(a.T).T[0:2, :])

units = rt['resp'].chans
e=rt['resp'].epochs
r_active = rt.copy().and_mask(['HIT_TRIAL','CORRECT_REJECT_TRIAL', 'MISS_TRIAL'])
r_passive = rt.copy().and_mask(['PASSIVE_EXPERIMENT'])
#r_miss = rt.copy().and_mask(['MISS_TRIAL'])
stim_len = int(0.5*rt['resp'].fs)

conditions = ['passive', 'active']
cond_recs = [r_passive, r_active]

from nems_lbhb.tin_helpers import make_tbp_colormaps, compute_ellipse
cmaps=make_tbp_colormaps(ref_stim, tar_stim)
siglist = ['ppc_pred', 'ppc_indep', 'ppc_lv', 'rpc']
f,ax=plt.subplots(len(conditions),len(siglist),sharex=True,sharey=True, figsize=(2*len(siglist),4))
for ci, to, r in zip(range(len(conditions)), conditions, cond_recs):
    for j, sig in enumerate(siglist):
        colors = cmaps[0]
        for i,k in enumerate(ref_stim):
            try:
                p = r[sig].extract_epoch(k, mask=r['mask'])
                if p.shape[0]>2:
                    g = np.isfinite(p[:,0,onset])
                    x = np.nanmean(p[g,0,onset:offset], axis=1)
                    y = np.nanmean(p[g,1,onset:offset], axis=1)
                    c=list(colors(i))
                    c[-1]=0.2
                    ax[ci, j].plot(x,y,'.', color=c, label=k)
                    e = compute_ellipse(x, y)
                    ax[ci, j].plot(e[0], e[1],color=colors(i))
            except:
                #print(f'no matches for {k}')
                pass

        colors = cmaps[1]
        for i,k in enumerate(tar_stim):
            try:
                p = r[sig].extract_epoch(k, mask=r['mask'])
                if p.shape[0]>2:
                    g = np.isfinite(p[:,0,onset])
                    x = np.nanmean(p[g,0,onset:offset], axis=1)
                    y = np.nanmean(p[g,1,onset:offset], axis=1)
                    c=list(colors(i))
                    c[-1]=0.2
                    ax[ci, j].plot(x,y,'.', color=c, label=k)
                    e = compute_ellipse(x, y)
                    ax[ci, j].plot(e[0], e[1],color=colors(i))
            except:
                #print(f'no matches for {k}')
                pass
        ax[ci,j].set_title(f"{to}-{sig}")

ax[0,0].set_ylabel(siteid)
ax[1,0].set_xlabel('PC1')
ax[1,0].set_ylabel('PC2')

if savefigs:
    outfile = f"pop_latent_sim_{siteid}_{batch}.pdf"
    print(f"saving to {outpath}/{outfile}")
    f.savefig(f"{outpath}/{outfile}")

recfile=f"{recpath}/{cellid}_{batch}_LV_sim_rec.tgz"
rt.save(recfile)

