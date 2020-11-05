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

rec_a = rec.and_mask(['HIT_TRIAL', 'MISS_TRIAL', 'CORRECT_REJECT_TRIAL'])

ra = rec['resp'].extract_epoch(ep, mask=rec_a['mask'])[:, :, start:end].mean(axis=-1)

# look for HIT / MISS dependent changes in PC / dDR space
