import json
import pandas as pd
from pathlib import Path
from signalforge.runtime import file_hash
from signalforge.cftc_integration import load_contract,release_evidence

REPO=Path(__file__).parents[2]


def test_registered_asset_mapping_codes_and_claim_boundary():
    contract,path=load_contract(REPO)
    assert set(contract['assets'])=={'SPY','QQQ','IEF','TLT','GLD','SLV','USO','UNG'}
    assert contract['assets']['SPY']['contract_code']=='13874A'
    assert contract['assets']['USO']['contract_code']=='067651'
    assert contract['assets']['UNG']['contract_code']=='023651'
    assert contract['positioning_is_cash_flow'] is False and contract['pit_tier']=='B'
    assert len(file_hash(path))==64


def test_release_reconstruction_blocks_shutdown_and_uses_known_ion_dates():
    contract,path=load_contract(REPO);h=file_hash(path)
    dates=['2018-12-18','2018-12-24','2019-03-26','2023-01-31','2023-02-07']
    clocks,blocked=release_evidence(dates,contract,h)
    assert '2018-12-24' not in clocks and any(x['report_date']=='2018-12-24' for x in blocked)
    assert pd.Timestamp(clocks['2018-12-18']['available_at'])==pd.Timestamp('2018-12-25T23:59:59',tz='America/New_York').tz_convert('UTC')
    assert pd.Timestamp(clocks['2019-03-26']['available_at'])==pd.Timestamp('2019-04-04T00:00:00-04:00').tz_convert('UTC')
    assert pd.Timestamp(clocks['2023-01-31']['available_at'])==pd.Timestamp('2023-02-24T15:30:00-05:00').tz_convert('UTC')
    assert pd.Timestamp(clocks['2023-02-07']['available_at'])==pd.Timestamp('2023-03-03T15:30:00-05:00').tz_convert('UTC')
