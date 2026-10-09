import pytest
from p1_pit_readiness_v34.gates import *
def test_schedule_not_clock():assert not certify({'payload_sha256':'a'},{'clock_kind':'scheduled','first_public_at':'2019'})
def test_edgar_acceptance_not_public():assert not certify({'payload_sha256':'a'},{'clock_kind':'accepted'})
def test_document_not_version():assert not certify({'payload_sha256':'a'},{'first_public_at':'2019','publisher':'official'})
def test_hash_mismatch():assert not certify({'payload_sha256':'a'},{'payload_sha256':'b','first_public_at':'2019','publisher':'official','dissemination_record':'r','version_link':'x','correction_history_complete':True,'clock_kind':'authenticated_first_public'})
def test_403_no_retry():assert not retrieval_allowed('HTTP_403_ENTITLEMENT')
def test_auth_no_fabrication():assert not retrieval_allowed('BLOCKED_AUTH')
def test_raw_open_not_p1():assert not p1_gate({'raw_open':True})
def test_future_no_fake_schedule():assert future_probe()['schedule'] is None
def test_used_outer_not_future():assert future_probe(cohort_dates=104)['state']=='BLOCKED_DATA_OR_FUTURE'
def test_authorization_not_freeze():assert not final_gate(True,False,False)
def test_freeze_not_strict_pit():assert not final_gate(True,True,False)
def test_typed_freeze_clock():
    with pytest.raises(ValueError):future_probe(operational_freeze='2026-10-08')
def test_actual_original_p1_receipt_daily_schema():
    import json,pathlib
    p=pathlib.Path('/home/USERNAME/.local/share/signalforge-qx-v3/da768f7446b1/artifacts/tiingo_inputs/ba9be84158dedc267ba5191b298084ffa5cdc4e7dbe92bb105e828f7dd4062f0/p1_preparation.json')
    d=json.loads(p.read_text());assert 'daily' in d and 'daily_rows' not in d and len(d['daily'])==27088
