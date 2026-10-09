import json
from urllib.parse import urlencode
import pytest
from signalforge.tiingo_actions import new_plan,validate_plan,validate_action_url,request,parse_actions


def test_action_plan_and_urls_are_bounded():
    plan=new_plan();assert validate_plan(plan)
    for kind in ['distributions','splits']:
        url,params=request('SPY',kind,plan)
        assert validate_action_url(url+'?'+urlencode(params))==('SPY',kind)
    with pytest.raises(PermissionError):
        validate_action_url('https://api.tiingo.com/tiingo/corporate-actions/SPY/distributions')


def test_distribution_parser_preserves_missing_optional_dates(tmp_path):
    plan=new_plan();path=tmp_path/'dist.json'
    path.write_text(json.dumps([{'ticker':'SPY','exDate':'2023-03-17T00:00:00Z','paymentDate':'2023-04-28T00:00:00Z',
        'recordDate':'2023-03-20T00:00:00Z','declarationDate':'2023-01-24T00:00:00Z','distribution':1.5,'distributionFrequency':'q'},
        {'ticker':'SPY','exDate':'2023-06-16T00:00:00Z','paymentDate':None,'recordDate':None,'declarationDate':None,
        'distribution':1.6,'distributionFrequency':'q'}]))
    rows=parse_actions(path,'SPY','distributions',plan)
    assert rows[0]['paymentDate'].startswith('2023-04-28') and rows[1]['paymentDate'] is None
    assert rows[0]['distribution']==1.5


def test_split_parser_retains_cancelled_status_and_ratio(tmp_path):
    path=tmp_path/'split.json';path.write_text(json.dumps([{'ticker':'QQQ','exDate':'2023-08-25T00:00:00Z',
        'splitFrom':1,'splitTo':2,'splitFactor':2,'splitStatus':'c'}]))
    row=parse_actions(path,'QQQ','splits')[0]
    assert row['splitStatus']=='c' and row['splitFactor']==2


def test_action_parser_rejects_reserved_date_and_bad_ticker(tmp_path):
    path=tmp_path/'bad.json';path.write_text(json.dumps([{'ticker':'SPY','exDate':'2024-01-02T00:00:00Z',
        'distribution':1.0,'paymentDate':None,'recordDate':None,'declarationDate':None,'distributionFrequency':'q'}]))
    with pytest.raises(PermissionError):parse_actions(path,'SPY','distributions')
    path.write_text(json.dumps([{'ticker':'QQQ','exDate':'2023-01-02T00:00:00Z','distribution':1.0}]))
    with pytest.raises(ValueError,match='ticker'):parse_actions(path,'SPY','distributions')
