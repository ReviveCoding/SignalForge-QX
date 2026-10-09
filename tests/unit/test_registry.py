import numpy as np
import pandas as pd
import pytest
from signalforge.experiment_registry import ExperimentSpec,apply_release_outage,training_weights,scale_weight_grid,invalid_control
from signalforge.ssl import validate_pretrain


def test_ablation_and_transport_types_separate():
    for intervention,retrained in [('retrained_ablation',True),('frozen_input',False),('explanation',False),('transport_zero_shot',False),('transport_refit',True)]:
        assert ExperimentSpec('E03' if 'transport' not in intervention else 'E06','Main-A','development',intervention,retrained,{}).validate()
    with pytest.raises(ValueError):ExperimentSpec('E03','Main-A','development','frozen_input',True,{}).validate()


def test_release_block_corruption_closure():
    events=pd.DataFrame({'source':['eia','eia','cftc'],'available_at':['2020-01-01T00:00Z','2020-01-08T00:00Z','2020-01-08T00:00Z'],'value':[1.,2.,3.]})
    schedule=[{'source':'eia','start':'2020-01-01T00:00Z','end':'2020-01-09T00:00Z','kind':'delay','days':14}]
    out,invalid,trace=apply_release_outage(events,schedule,{'change':['eia'],'surprise':['change'],'gate':['surprise']})
    assert out.iloc[0].available_at==pd.Timestamp('2020-01-15T00:00Z') and trace[0]['affected_release_rows']==2
    assert invalid==['change','eia','gate','surprise']


def test_scale_grid_frozen_small_rule_and_train_weights():
    rows=scale_weight_grid({'partition':'inner_validation','frozen_rule_hash':'fixture'})
    assert len(rows)==54 and len({r['small_rule_hash'] for r in rows})==1
    with pytest.raises(PermissionError):scale_weight_grid({'partition':'outer','frozen_rule_hash':'fixture'})
    dates=pd.date_range('2016-01-01',periods=30,freq='7D',tz='UTC')
    for rule in ['uniform','recency','lagged_vol_regime_balanced']:
        weights=training_weights(rule,dates,np.linspace(0,1,30),dates[-1])
        assert (weights>0).all() and weights.mean()==pytest.approx(1)


def test_invalid_controls_never_promoted():
    events=pd.DataFrame({'entity':['x'],'source':['eia'],'field':['stocks'],'reference_time':['2020-01-01T00:00Z'],'available_at':['2020-01-08T00:00Z'],'value':[1.]})
    out=invalid_control(events,'wrong_release','development')
    assert out.invalid_control.all() and not out.promotion_eligible.any() and out.pit_tier.eq('C').all()
    with pytest.raises(PermissionError):invalid_control(events,'wrong_release','reserved')


def test_ssl_rejects_future_lineage_before_cuda():
    x=np.ones((2,3,4));dates=['2020-01-01T00:00Z','2020-01-08T00:00Z']
    with pytest.raises(PermissionError):validate_pretrain(x,dates,dates,'2020-01-05T00:00Z')
    with pytest.raises(PermissionError):validate_pretrain(x,dates,['2020-01-02T00:00Z',dates[1]],'2020-01-09T00:00Z')
