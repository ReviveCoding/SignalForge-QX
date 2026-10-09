import numpy as np
import pandas as pd
from signalforge.diagnostics import SERIES,weekly_feature,rebuild_physical_features


def test_future_release_cannot_change_frozen_origin_features():
    origin=pd.Timestamp('2020-01-10T23:00Z')
    old={'available_at':pd.Timestamp('2020-01-09T05:00Z'),'reference_time':pd.Timestamp('2020-01-03T00:00Z'),'raw_hash':'old','values':np.arange(10.)}
    future={**old,'available_at':pd.Timestamp('2020-01-16T05:00Z'),'reference_time':pd.Timestamp('2020-01-10T00:00Z'),'values':np.full(10,999999.)}
    np.testing.assert_array_equal(weekly_feature(origin,[old])[0],weekly_feature(origin,[old,future])[0])


def test_release_outage_rebuild_keeps_targets_and_real_stale_age():
    rows=[]
    for hash_,available,reference,value in [('old','2020-01-02T05:00Z','2019-12-27T00:00Z',1.),('blocked','2020-01-09T05:00Z','2020-01-03T00:00Z',np.nan)]:
        for name in SERIES:
            for field in ['value','change']:
                rows.append({'raw_hash':hash_,'available_at':available,'reference_time':reference,'field':name+'_'+field,'value':value})
    original=pd.DataFrame({'decision_time':['2020-01-10T23:00Z'],'y':[17.],'label_available_at':['2020-01-16T05:00Z'],'x':[[0.]*14],'raw_hash':['blocked']})
    rebuilt=rebuild_physical_features(original,pd.DataFrame(rows))
    assert rebuilt.y.tolist()==original.y.tolist() and rebuilt.label_available_at.tolist()==original.label_available_at.tolist()
    assert rebuilt.raw_hash.iloc[0]=='old' and rebuilt.x.iloc[0][10]==8.75
    assert original.raw_hash.iloc[0]=='blocked' and original.x.iloc[0][10]==0.
