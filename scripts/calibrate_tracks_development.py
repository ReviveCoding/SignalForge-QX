"""ADR-014 development calibration of explicit immutable existing seed ensembles.

These are frozen outer-2022 members, not newly selected final candidates.
The fit cutoff and this limitation are recorded; no reserved data is opened.
"""
import json
import argparse
import io
from contextlib import nullcontext
import numpy as np
import pandas as pd
from signalforge.runtime import paths,atomic_json,digest,commit_bundle,validate_bundle,gpu_lease,now,file_hash
from signalforge.successor_inputs import prepared_track
from signalforge.successor import require_successor_access,ledger_snapshot,assert_parent_unchanged
from signalforge.experiments import ComputeBudget
from signalforge.inference import infer_candidates
from signalforge.frozen_processing import candidate_ensemble_id
from signalforge.postprocess import Calibrator
from signalforge.models import QUANTILES
from signalforge.track_engine import score
from signalforge.successor_runtime import progress
from signalforge.successor_backend import activate_backend

repo,runtime=paths();study=json.loads((repo/'configs/study.json').read_text());minimum=json.loads((repo/'configs/statistical_contract.json').read_text())['calibration']['minimum_distinct_dates']
activate_backend(repo,runtime)
parser=argparse.ArgumentParser();parser.add_argument('--track',choices=['Main-A','Nested-B','both'],default='both');options=parser.parse_args()
q=json.loads((repo/'reports/track_input_qualification_v311.json').read_text());full=repo/'reports/successor_gpu_full.json'
rows={t['track']:list(t['cpu_development']['results']) for t in q['tracks']}
if full.exists():
    for t in json.loads(full.read_text())['tracks']:rows[t['track']].extend(t['results'])
parent=ledger_snapshot(runtime/'ledger/development_compute.sqlite',43200)
for track,results in rows.items():
    if options.track!='both' and track!=options.track:continue
    panels,contexts,qualification=prepared_track(repo,runtime,track)
    candidates=[]
    for info,family in sorted({(r['information'],r['family']) for r in results if r.get('year')==2022 and r['state']=='SUCCEEDED'}):
        chosen=[r for r in results if r.get('year')==2022 and r['information']==info and r['family']==family and r['state']=='SUCCEEDED']
        if sorted(r['seed'] for r in chosen)!=[11,37,71]:raise ValueError('Complete three-seed development ensemble required')
        members=[]
        for r in sorted(chosen,key=lambda r:r['seed']):
            receipt=validate_bundle(runtime/r['model_relative_bundle'])
            if digest(receipt)!=r['model_receipt_id']:raise PermissionError('Immutable model mismatch')
            members.append({'member_id':str(r['seed']),'relative_bundle':r['model_relative_bundle'],'receipt_id':r['model_receipt_id']})
        candidates.append({'track':track,'candidate_id':track+'_'+info+'_'+family+'_outer2022_frozen','family':family,'information':info,
                           'members':members,'weights':[1/3]*3,'normalizer_id':chosen[0]['normalizer_id']})
    family_pointer={'state':'FROZEN_DEVELOPMENT_CALIBRATION_CANDIDATES','track':track,'candidates':candidates,'reserved_access':False}
    frozen=runtime/'artifacts/track_calibration_candidates'/digest(family_pointer)
    commit_bundle(frozen,{'candidates.json':family_pointer},{'track':track,'reserved_access':False})
    receipts=[]
    for candidate in candidates:
        progress(repo,runtime,'TRACK CALIBRATION',track=track,family=candidate['family'],information_set=candidate['information'],
                 completed_candidates=len(receipts),planned_candidates=len(candidates),status='CALIBRATING_FROZEN_DEVELOPMENT_ENSEMBLE')
        info=candidate['information'];panel=panels[info];dates=pd.to_datetime(panel.decision_time,utc=True);available=pd.to_datetime(panel.label_available_at,utc=True)
        window=panel.loc[dates.ge('2023-01-01T00:00Z')&dates.lt('2024-01-01T00:00Z')&available.le('2023-12-31T23:59:59Z')&panel.y.notna()].copy()
        if window.empty:raise RuntimeError('SCIENTIFIC_OR_DATA_GATE: no mature selection/calibration rows')
        raw=contexts[info][window.sequence_index];identity=digest({'candidate':candidate,'grid':window[['decision_time','asset']].astype(str).to_dict('records')})
        directory=runtime/'artifacts/track_calibration_predictions'/identity
        if (directory/'receipt.json').exists():
            validate_bundle(directory);saved=np.load(directory/'predictions.npz');mean=saved['mean'];quantiles=saved['quantiles']
        else:
            gpu=candidate['family'] not in {'historical','ewma','ridge','linear_quantile','mixed_frequency_shrinkage'}
            budget=ComputeBudget(runtime/'ledger/successor_gpu_full_compute.sqlite',21600) if gpu else None
            try:
                if gpu:require_successor_access(repo)
                with gpu_lease(runtime) if gpu else nullcontext():
                    if gpu:
                        from signalforge.resources import require_current_resources
                        require_current_resources(repo,runtime)
                    with budget.charge('calibration_inference_'+identity,300) if gpu else nullcontext():
                        mean,quantiles=infer_candidates([candidate],raw,runtime,assets=window.asset.tolist(),context_eligible=window.context_eligible.to_numpy(dtype=bool))[candidate['candidate_id']]
            finally:
                if budget:budget.close()
            buffer=io.BytesIO();np.savez(buffer,mean=mean,quantiles=quantiles)
            commit_bundle(directory,{'predictions.npz':buffer.getvalue(),'grid.json':window[['decision_time','asset','label_available_at']].astype(str).to_dict('records')},
                          {'track':track,'candidate_id':candidate['candidate_id'],'ensemble_id':candidate_ensemble_id(candidate),'reserved_access':False})
        scales=json.loads((runtime/candidate['members'][0]['relative_bundle']/'asset_scales.json').read_text())
        scale=window.asset.map(scales).to_numpy();cal_mask=pd.to_datetime(window.decision_time,utc=True).ge('2023-07-01T00:00Z').to_numpy()
        cal_rows=window.loc[cal_mask];ensemble=candidate_ensemble_id(candidate)
        calibrator=Calibrator().fit(cal_rows.y.to_numpy()/scale[cal_mask],quantiles[cal_mask]/scale[cal_mask,None],cal_rows.decision_time,
            cal_rows.label_available_at,ensemble,'2023-07-01T00:00Z','2023-12-31T23:59:59Z',minimum)
        processed=calibrator.transform(quantiles[cal_mask]/scale[cal_mask,None],ensemble)['processed']*scale[cal_mask,None]
        manifests=[json.loads((runtime/m['relative_bundle']/'transform.json').read_text()) for m in candidate['members']]
        receipts.append({'candidate':candidate,'ensemble_id':ensemble,'calibration':calibrator.manifest(),
                         'model_fit_cutoffs':[m['fit_cutoff'] for m in manifests],
                         'prediction_hash':file_hash(directory/'predictions.npz'),'prediction_relative_bundle':str(directory.relative_to(runtime)),
                         'raw_calibration_score':score(cal_rows,mean[cal_mask],quantiles[cal_mask],scale[cal_mask]),
                         'calibrated_in_sample_score':score(cal_rows,mean[cal_mask],processed,scale[cal_mask]),
                         'selection_score':score(window.loc[~cal_mask],mean[~cal_mask],quantiles[~cal_mask],scale[~cal_mask]),
                         'source_consistency':'VERIFIED_HASH_BOUND_RECONSTRUCTED_P0_PANEL',
                         'original_publication_qualified':False,'qualified_for_freeze':False})
        assert_parent_unchanged(runtime,parent)
    report={'state':'COMPLETED_DEVELOPMENT_CALIBRATION_TIER_B','track':track,'created_at':now(),'candidates':receipts,
            'candidate_family_id':digest(family_pointer),'support_unit':'independent mature decision dates, not asset rows or seeds',
            'claim_boundary':'existing outer-2022 three-seed development ensembles; not five-seed final candidates; calibration-window effects are in-sample diagnostics',
            'original_publication_qualified':False,'qualified_for_freeze':False,'qualified_for_final':False,'reserved_access':False}
    atomic_json(repo/'reports'/('calibration_'+track.replace('-','_')+'.json'),report)
    print(json.dumps({'track':track,'state':report['state'],'candidates':len(receipts)}),flush=True)
