"""
Dump dprime results using cross-validation 

Steps:
    * Load data (get all stim tags, a/p masks etc.)
    * Generate list of stimulus pairs
    * Generate est / val sets for each stimulus pair
    * Compute dprime for each stim pair
"""
from nems_lbhb.baphy_experiment import BAPHYExperiment
from decoding_loader import load_behavior_dataset

# developing code, these will come from sys arguments eventually, so that it can run on 
# the cluster
batch = 324
site = 'CRD018d'

options = {'resp': True, 'pupil': True, 'rasterfs': 10}

# Load the data
manager = BAPHYExperiment(batch=batch, siteid=site[:7])

dataset = load_behavior_dataset(manager, **options)