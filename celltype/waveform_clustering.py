"""
Cluster spike waveforms for batches 324 / 325
 - Using peak / trough ratio and spike width
Show clustering + mean waveforms per group
"""
import nems.db as nd
import nems_lbhb.baphy_io as io
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 6

recache = False
batches = [324, 325]

cellids = []
for b in batches:
    cellids += nd.get_batch_cells(batch=b).cellid.values.tolist()

# get mean waveform for each cellid
if recache:
    mwf = []
    for c in cellids:
        if 'CRD' in c:
            animal = 'Cordyceps'
        elif 'ARM' in c:
            animal = 'Armillaria'
        mwf.append(io.get_mean_spike_waveform(cellid=c, animal=animal))
    
    mwf = np.stack(mwf)
    mwf = pd.DataFrame(index=cellids, data=mwf)
    mwf.to_pickle(DIR + '/results/mwf_df.pickle')

    # get waveform stats from database
    sw = [nd.get_gSingleCell_meta(cellid=c, fields='wft_spike_width') for c in cellids] 
    pt = [nd.get_gSingleCell_meta(cellid=c, fields='wft_peak_trough_ratio') for c in cellids] 
    wft_stats = pd.DataFrame(index=cellids, columns=['spike_width', 'peak_trough'], data=np.stack([sw, pt]).T)
    iso_query = f"SELECT cellid, min_isolation from Batches WHERE cellid in {tuple([x for x in cellids])}"
    isolation = nd.pd_query(iso_query)
    wft_stats = pd.merge(wft_stats, isolation, left_index=True, right_on='cellid')
    wft_stats.to_pickle(DIR + '/results/wft_stats.pickle')

else:
    mwf = pd.read_pickle(DIR + '/results/mwf_df.pickle')
    wft_stats = pd.read_pickle(DIR + '/results/wft_stats.pickle')

# cluster / plot mean waveforms for single units
stats = wft_stats[wft_stats.min_isolation>95]
min_iso = 95
iso_mask = stats.min_isolation>min_iso

clust_stats = stats[['spike_width', 'peak_trough']].copy()
clust_stats -= clust_stats.mean(axis=0)
clust_stats /= clust_stats.std(axis=0)
weights = [.1 if ((stats['spike_width'].iloc[i]>0.3) & (stats['spike_width'].iloc[i]<0.3)) else 1 for i in range(stats.shape[0])]
km = KMeans(n_clusters=2, init=np.array([[0.6, 0.25], [0.15, 0.5]])).fit(stats[['spike_width', 'peak_trough']])
stats.at[:, 'type'] = km.labels_

ms = 10

f, ax = plt.subplots(1, 2, figsize=(4, 2))

ax[0].scatter(stats[(stats.type==1) & iso_mask]['spike_width'],
            stats[(stats.type==1) & iso_mask]['peak_trough'], s=ms)

ax[0].scatter(stats[(stats.type==0) & iso_mask]['spike_width'],
            stats[(stats.type==0) & iso_mask]['peak_trough'], s=ms)

ax[0].scatter(stats[(stats.type==1) & ~iso_mask]['spike_width'],
            stats[(stats.type==1) & ~iso_mask]['peak_trough'], s=ms, color='tab:blue', alpha=0.3)

ax[0].scatter(stats[(stats.type==0) & ~iso_mask]['spike_width'],
            stats[(stats.type==0) & ~iso_mask]['peak_trough'], s=ms, color='tab:orange', alpha=0.3)

ax[0].set_xlabel("Spike Width (ms)")
ax[0].set_ylabel("Peak-trough ratio")

# get mean of each cluster
g1 = stats[stats.type==1].cellid
ax[1].plot(mwf.loc[g1].mean(axis=0), lw=1)
ax[1].fill_between(range(0, mwf.shape[1]),
        mwf.loc[g1].mean(axis=0) - mwf.loc[g1].sem(axis=0),
        mwf.loc[g1].mean(axis=0) + mwf.loc[g1].sem(axis=0), 
        alpha=0.5
    )
g2 = stats[stats.type==0].cellid
ax[1].plot(mwf.loc[g2].mean(axis=0), lw=1)
ax[1].fill_between(range(0, mwf.shape[1]),
        mwf.loc[g2].mean(axis=0) - mwf.loc[g2].sem(axis=0),
        mwf.loc[g2].mean(axis=0) + mwf.loc[g2].sem(axis=0), 
        alpha=0.5
    )

f.tight_layout()

plt.show()
