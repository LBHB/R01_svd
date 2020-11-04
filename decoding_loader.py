
def load_behavior_dataset(manager, twin=0.2, psth_sim=False, ind_sim=False, lv_sim=False,
                                             regress_pupil=False, regress_behavior=False, sim_first_order=False)
"""
Helper function to load TBP / PTD / BVT files for dprime analysis.

    Input:
        manager: baphy experiment manager
        twin: twindow to collapse over for decoding, relative to sound onset. 
                e.g. twin = 0.2 means use first 200 ms of sound
        booleans indicate whether to regress out stats, load simulation, etc.

    Return: 
        dataset (a dictionary) containing the following:
            resp:  dictionary of responses
            amask: dictionary of active mask
            pmask: dictionary of passive mask
            tar:   list of targets
            cat:   list of catches
            ref:   list of ref_stim
"""