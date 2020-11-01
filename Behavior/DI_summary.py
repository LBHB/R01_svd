from settings import DIR
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 6

savefig = True

arm_di = pickle.load(open(DIR+'/results/Armillaria_DI.pickle', 'rb'))
crd_di = pickle.load(open(DIR+'/results/Cordyceps_DI.pickle', 'rb'))

snrs = ['-5', '0', 'inf']
df = pd.DataFrame()
for siteid, site in zip(['ARM', 'CRD'], [arm_di, crd_di]):
    for snr in snrs:
        data = np.stack([[siteid]*len(site[snr]), site[snr], [snr]*len(site[snr])])
        df = df.append(pd.DataFrame(columns=['site', 'di', 'snr'], data=data.T))

dtypes = {'di': 'float32','site': 'object', 'snr': 'category'}
df = df.astype(dtypes)


f, ax = plt.subplots(1, 1, figsize=(2, 2))

ax.errorbar([0, 1, 2], df[df.site=='CRD'][['di', 'snr']].groupby(by='snr').mean()['di'], 
                          yerr=df[df.site=='CRD'][['di', 'snr']].groupby(by='snr').sem()['di'], 
                          marker='o', capsize=3, lw=1, markeredgewidth=1, label='CRD')
ax.errorbar([0, 1, 2], df[df.site=='ARM'][['di', 'snr']].groupby(by='snr').mean()['di'], 
                          yerr=df[df.site=='CRD'][['di', 'snr']].groupby(by='snr').sem()['di'], 
                          marker='o', capsize=3, lw=1, markeredgewidth=1, label='ARM')
ax.axhline(0.5, linestyle='--', color='grey', lw=1)
ax.legend(frameon=False)
ax.set_ylabel('DI')
ax.set_xlabel('SNR')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['-5', '0', 'Inf'])

f.tight_layout()

if savefig:
    f.savefig(DIR + '/results/figures/DI_summary.pdf')

plt.show()