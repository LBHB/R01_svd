from nems_lbhb.baphy import parse_cellid
import nems_lbhb.tin_helpers as thelp
import numpy as np
import charlieTools.preprocessing as preproc

def load_behavior_dataset(manager, recache=False, ts=0.2, te=0.4, psth_sim=False, ind_sim=False, lv_sim=False,
                                             regress_pupil=False, regress_task=False, sim_first_order=False, **options):
    """
    Helper function to load TBP / PTD / BVT files for dprime analysis.

        Input:
            manager: baphy experiment manager
            ts/te: time start/end to collapse over for decoding, relative to epoch onset (PreStimSilence onset). 
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

    # empty dictionary to store results
    dataset = {}

    batch = manager.batch
    rec = manager.get_recording(recache=recache, **options)
    rec['resp'] = rec['resp'].rasterize()
    if batch == 302:
        c, _ = parse_cellid({'cellid': manager.siteid, 'batch': batch})
        rec['resp'] = rec['resp'].extract_channels(c)
        
    # mask appropriate trials
    if batch in [324, 325]:
        active_mask = ['HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'MISS_TRIAL']
        rec = rec.and_mask(['PASSIVE_EXPERIMENT', 'HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'MISS_TRIAL'])
    elif batch == 307:
        active_mask = ['HIT_TRIAL']
        rec = rec.and_mask(['PASSIVE_EXPERIMENT', 'HIT_TRIAL'])
    elif batch == 302:
        active_mask = ['HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'INCORRECT_HIT_TRIAL']
        rec = rec.and_mask(['PASSIVE_EXPERIMENT', 'HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'INCORRECT_HIT_TRIAL'])

    rec = rec.apply_mask(reset_epochs=True)

    # state corrections 
    if regress_pupil & regress_task:
        rec = preproc.regress_state(rec, state_sigs=['pupil', 'behavior'])
    elif regress_pupil:
        rec = preproc.regress_state(rec, state_sigs=['pupil'])
    elif regress_task:
        rec = preproc.regress_state(rec, state_sigs=['behavior'])

    # ================================== get active / passive masks ==================================
    ra = rec.copy()
    ra = ra.create_mask(True)
    ra = ra.and_mask(active_mask)

    rp = rec.copy()
    rp = rp.create_mask(True)
    rp = rp.and_mask(['PASSIVE_EXPERIMENT'])

    _rp = rp.apply_mask(reset_epochs=True)
    _ra = ra.apply_mask(reset_epochs=True)

    # =================================== find / sort epoch names ====================================
    # need to do some "hacky" stuff for batch 302 / 307 to get names to align with the TIN data
    if batch in [324, 325]:
        targets = thelp.sort_targets([f for f in _ra['resp'].epochs.name.unique() if 'TAR_' in f])
        # only keep target presented at least 5 times
        targets = [t for t in targets if (_ra['resp'].epochs.name==t).sum()>=5]
        # remove "off-center targets"
        on_center = thelp.get_tar_freqs([f.strip('REM_') for f in _ra['resp'].epochs.name.unique() if 'REM_' in f])[0]
        targets = [t for t in targets if str(on_center) in t]
        if len(targets)==0:
            # NOT ENOUGH REPS AT THIS SITE
            skip_site = True
        catch = [f for f in _ra['resp'].epochs.name.unique() if 'CAT_' in f]
        # remove off-center catches
        catch = [c for c in catch if str(on_center) in c]
        rem = [f for f in rec['resp'].epochs.name.unique() if 'REM_' in f]
        targets_str = targets
        catch_str = catch
        ref_stim = thelp.sort_refs([f for f in _ra['resp'].epochs.name.unique() if 'STIM_' in f])
        ref_str = ref_stim
        tar_idx = 0
    elif batch == 307:
        params = manager.get_baphy_exptparams()
        params = [p for p in params if p['BehaveObjectClass']!='Passive'][0]
        tf = params['TrialObject'][1]['TargetHandle'][1]['Names']
        targets = [f'TAR_{t}' for t in tf]
        if params['TrialObject'][1]['OverlapRefTar']=='Yes':
            snrs = params['TrialObject'][1]['RelativeTarRefdB'] 
        else:
            snrs = ['Inf']
        snrs = [s if (s!=np.inf) else 'Inf' for s in snrs]
        #catchidx = int(params['TrialObject'][1]['OverlapRefIdx'])
        refs = params['TrialObject'][1]['ReferenceHandle'][1]['Names']
        catch = ['REFERENCE'] #['STIM_'+refs[catchidx]]
        catch_str = [f'CAT_{tf[0]}+-InfdB+Noise+allREFs']
        targets_str = [f'TAR_{t}+{snr}dB+Noise' for snr, t in zip(snrs, tf)]
        targets_str = targets_str[::-1]
        targets = targets[::-1]

        # only keep targets w/ at least 5 reps in active
        targets_str = [ts for t, ts in zip(targets, targets_str) if (_ra['resp'].epochs.name==t).sum()>=5]
        targets = [t for t in targets if (_ra['resp'].epochs.name==t).sum()>=5]
        
        ref_stim = [f for f in _ra['resp'].epochs.name.unique() if 'STIM_' in f]
        ref_str = [f"STIM_{tf[0]}+torc{r.split('LIN_')[1].split('_v')[0]}" for r in ref_stim]

        # only keep refs with at least 3 reps
        ref_str = [ts for t, ts in zip(ref_stim, ref_str) if (_ra['resp'].epochs.name==t).sum()>=3]
        ref_stim = [t for t in ref_stim if (_ra['resp'].epochs.name==t).sum()>=3]

        tar_idx = 0

    elif batch == 302:
        params = manager.get_baphy_exptparams()
        params = [p for p in params if p['BehaveObjectClass']!='Passive'][0]
        tf = params['TrialObject'][1]['TargetHandle'][1]['Names']
        targets = [f'TAR_{t}' for t in tf]
        pdur = params['BehaveObject'][1]['PumpDuration']
        rew = np.array(tf)[np.array(pdur)==1].tolist()
        catch = [t for t in targets if (t.split('TAR_')[1] not in rew)]
        catch_str = [(t+'+InfdB+Noise').replace('TAR_', 'CAT_') for t in targets if (t.split('TAR_')[1] not in rew)]
        targets = [t for t in targets if (t.split('TAR_')[1] in rew)]
        targets_str = [t+'+InfdB+Noise' for t in targets if (t.split('TAR_')[1] in rew)]
        ref_stim = thelp.sort_refs([f for f in _ra['resp'].epochs.name.unique() if 'STIM_' in f])
        ref_str = ref_stim
        tar_idx = 1

    dataset['tar'] = targets
    dataset['ref'] = ref_stim
    dataset['cat'] = catch

    all_stim = targets + catch + ref_stim

    # extract spike counts and masks and bin according to twin
    sidx = int(rec['resp'].fs * ts)
    eidx = int(rec['resp'].fs * te)

    spikes = rec['resp'].extract_epochs(all_stim)
    spikes = {k: v[:, :, sidx:eidx].mean(axis=-1) for (k, v) in spikes.items()}

    amask = ra['mask'].extract_epochs(all_stim)
    amask = {k: v[:, :, 0] for (k, v) in amask.items()}

    pmask = rp['mask'].extract_epochs(all_stim)
    pmask = {k: v[:, :, 0] for (k, v) in pmask.items()}

    dataset['resp'] = spikes
    dataset['amask'] = amask
    dataset['pmask'] = pmask

    return dataset
