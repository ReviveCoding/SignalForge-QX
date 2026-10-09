import json
from pathlib import Path
import pytest
from signalforge.stage_cache import training_dependencies,record_training_dependencies,adopt_bundle,legacy_compatible
from signalforge.runtime import digest,code_hash,commit_bundle,validate_bundle


def repository(tmp_path):
    src=tmp_path/'src/signalforge';src.mkdir(parents=True)
    for name in ['data','features','models','neural','residual','ssl','pit','metrics','sources']:
        (src/(name+'.py')).write_text('VALUE=1\n')
    (src/'auxiliary.py').write_text('def sequences(): pass\ndef fit_predict(): pass\n')
    (src/'development.py').write_text('def preprocess(): pass\ndef trial_grid(): pass\ndef fit_artifact(): pass\ndef run_auxiliary():\n config={"epochs":100}\n for year in []: pass\n')
    (tmp_path/'configs').mkdir();(tmp_path/'reports').mkdir()
    (tmp_path/'configs/study.json').write_text(json.dumps({'splits':{'outer':[2018]},'hpo':{'seeds':[11]}}))
    (tmp_path/'reports/environment_audit.json').write_text(json.dumps({'resolved_packages':{'numpy':'fixture'},'python':'fixture','dispatcher_sha256':'fixture'}))
    return tmp_path


def test_reporting_postprocessing_do_not_invalidate_training(tmp_path):
    repo=repository(tmp_path);before=digest(training_dependencies(repo));whole=code_hash(repo)
    (repo/'src/signalforge/reporting.py').write_text('PLOT_STYLE="changed"\n')
    (repo/'src/signalforge/postprocess.py').write_text('CALIBRATION="changed"\n')
    assert code_hash(repo)!=whole and digest(training_dependencies(repo))==before
    (repo/'src/signalforge/models.py').write_text('VALUE=2\n')
    assert digest(training_dependencies(repo))!=before


@pytest.mark.parametrize('change',['grid','split','environment'])
def test_training_recipe_changes_invalidate_cache(tmp_path,change):
    repo=repository(tmp_path);before=digest(training_dependencies(repo))
    if change=='grid':
        p=repo/'src/signalforge/development.py';p.write_text(p.read_text().replace('def trial_grid(): pass','def trial_grid(): return [2]'))
    elif change=='split':
        p=repo/'src/signalforge/development.py';p.write_text(p.read_text().replace('for year in []: pass','for year in [2019]: pass'))
    else:
        p=repo/'reports/environment_audit.json';d=json.loads(p.read_text());d['resolved_packages']['numpy']='changed';p.write_text(json.dumps(d))
    assert digest(training_dependencies(repo))!=before


def test_verified_legacy_adoption_preserves_binary_and_source_receipt(tmp_path):
    repo=repository(tmp_path/'repo');runtime=tmp_path/'runtime'
    training=record_training_dependencies(repo,runtime);original=code_hash(repo)
    expected={'family':'ridge','seed':11,'trial':{'regularization':1.},'data_id':'fixture','normalizer_id':'fixture'}
    metric={**expected,'state':'SUCCEEDED','evidence_kind':'development','run_id':'old','seconds':1.,'synthetic_test_fixture':True}
    root=runtime/'artifacts/auxiliary'
    commit_bundle(root/'old',{'metrics.json':metric,'predictions.json':{'fixture':True},'transform.json':{'fixture':True},'model.bin':b'fixture-binary'},
                  {'code_hash':original,'data_id':'fixture','run_id':'old'})
    source=validate_bundle(root/'old')
    assert legacy_compatible(runtime,original,training)
    adopted=adopt_bundle(runtime,'old','new','report-code-changed',training,expected)
    assert adopted['cache_parent_run_id']=='old' and adopted['seconds']==1.
    assert (root/'new/model.bin').read_bytes()==b'fixture-binary'
    assert validate_bundle(root/'old')==source
    assert adopt_bundle(runtime,'old','new','another-report-change',training,expected)==adopted
    with pytest.raises(ValueError,match='recipe mismatch'):
        adopt_bundle(runtime,'old','bad','changed',training,{**expected,'seed':37})
    assert adopt_bundle(runtime,'old','incompatible','changed','wrong-training-implementation',expected) is None
