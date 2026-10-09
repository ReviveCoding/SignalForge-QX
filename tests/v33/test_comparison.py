import numpy as np,pandas as pd,pytest
from signalforge.v33_comparison import paired_comparison,COLS
def fixture():
    rows=[]
    for d in ['2021-01-01T00:00Z','2021-01-08T00:00Z']:
        for a in ['a','b']:
            for s in [11,37,71]:
                rows.append(dict(decision_time=d,asset=a,seed=s,target=0.,scale=1.,**dict(zip(COLS,[-2.,-1.,0.,1.,2.]))))
    f=pd.DataFrame(rows); meta=dict(fold_id='f',source_id='s',normalizer_id='n',target_contract_id='t')
    return f,meta
def test_equal_predictions_zero_gain_and_date_unit():
    f,m=fixture();out=paired_comparison(f,f.copy(),m,m)
    assert out['relative_pinball_gain']==0 and out['independent_dates']==2 and out['rows_dropped']==0
def test_keep_catastrophic_rows_and_no_quantile_mutation():
    f,m=fixture();g=f.copy();g.loc[0,COLS]=np.array([1,2,3,4,5])*1e6
    before=g.copy(deep=True);out=paired_comparison(f,g,m,m)
    assert out['new']['forecast_rows_abs_over100']==1 and out['relative_pinball_gain']<0
    pd.testing.assert_frame_equal(g,before)
@pytest.mark.parametrize('change',['drop','target','scale','seed'])
def test_parity_failures(change):
    f,m=fixture();g=f.copy()
    if change=='drop':g=g.iloc[:-1]
    elif change=='seed':g=g[g.seed!=71]
    else:g.loc[0,change]=999
    with pytest.raises(PermissionError):paired_comparison(f,g,m,m)
def test_metadata_mismatch_blocks():
    f,m=fixture();other={**m,'normalizer_id':'wrong'}
    with pytest.raises(PermissionError):paired_comparison(f,f,m,other)
def test_reserved_guard_before_score():
    f,m=fixture();f.decision_time='2024-01-01T00:00Z';f.target=np.nan
    with pytest.raises(PermissionError):paired_comparison(f,f,m,m)
