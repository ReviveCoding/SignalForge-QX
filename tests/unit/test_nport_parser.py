import zipfile
import numpy as np
import pandas as pd
import pytest
from signalforge.nport import parse_nport, month_snapshot


def fixture(tmp_path, missing=False, duplicate=False):
    path=tmp_path/'archive.zip'
    submission=pd.DataFrame([{'ACCESSION_NUMBER':'original','REPORT_DATE':'2020-03-31',
                             'REPORT_ENDING_PERIOD':'2020-12-31','FILING_DATE':'2020-04-29','SUB_TYPE':'NPORT-P'},
                            {'ACCESSION_NUMBER':'amended','REPORT_DATE':'2020-03-31',
                             'REPORT_ENDING_PERIOD':'2020-12-31','FILING_DATE':'2020-07-01','SUB_TYPE':'NPORT-P/A'}])
    fund=[]
    for accession,sales in [('original',100),('amended',300)]:
        row={'ACCESSION_NUMBER':accession,'SERIES_ID':'S001','NET_ASSETS':1000}
        for i in range(1,4):
            row.update({f'SALES_FLOW_MON{i}':sales+i,f'REDEMPTION_FLOW_MON{i}':50,
                        f'REINVESTMENT_FLOW_MON{i}':20})
        if missing:row['SALES_FLOW_MON1']=None
        fund.append(row)
    if duplicate:fund.append(fund[0])
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('SUBMISSION.tsv',submission.to_csv(sep='\t',index=False))
        z.writestr('FUND_REPORTED_INFO.tsv',pd.DataFrame(fund).to_csv(sep='\t',index=False))
    clocks=[{'ACCESSION_NUMBER':a,'available_at':d,'pit_tier':'B','clock_evidence':'fixture_dissemination'}
            for a,d in [('original','2020-06-01T00:00Z'),('amended','2020-08-01T00:00Z')]]
    return path,clocks


def test_reference_months_not_fiscal_end_or_zip_quarter(tmp_path):
    path,clocks=fixture(tmp_path);rows,blocked=parse_nport(path,clocks)
    assert not blocked and list(rows.reference_month.unique())==['2020-01','2020-02','2020-03']
    assert rows.iloc[0].external_flow==51 and rows.iloc[0].reinvestment==20
    assert not rows.organic_flow_qualified.any()


def test_amendment_only_after_publication_and_fixed_cohort(tmp_path):
    path,clocks=fixture(tmp_path);rows,_=parse_nport(path,clocks)
    before=month_snapshot(rows,'2020-01','2020-07-15T00:00Z',['S001'])
    after=month_snapshot(rows,'2020-01','2020-08-02T00:00Z',['S001'])
    assert before['external_flow']==51 and after['external_flow']==251
    missing=month_snapshot(rows,'2020-01','2020-08-02T00:00Z',['S001','S002'])
    assert missing['state']=='BLOCKED_DATA' and missing['external_flow'] is None
    early=month_snapshot(rows,'2020-01','2020-05-01T00:00Z',['S001'])
    assert early['state']=='BLOCKED_DATA'


def test_missing_flow_remains_missing(tmp_path):
    path,clocks=fixture(tmp_path,missing=True);rows,_=parse_nport(path,clocks)
    assert rows[rows.reference_month=='2020-01'].external_flow.isna().all()
    assert month_snapshot(rows,'2020-01','2020-09-01T00:00Z',['S001'])['external_flow'] is None


def test_missing_dissemination_not_filing_date(tmp_path):
    path,clocks=fixture(tmp_path);rows,blocked=parse_nport(path,clocks[:1])
    assert len(rows)==3 and blocked[0]['accession']=='amended'


def test_duplicate_key_and_false_tier_a_rejected(tmp_path):
    path,clocks=fixture(tmp_path,duplicate=True)
    with pytest.raises(ValueError,match='duplicate'):parse_nport(path,clocks)
    path,clocks=fixture(tmp_path);clocks[0]['pit_tier']='A'
    with pytest.raises(ValueError,match='Tier A'):parse_nport(path,clocks)


def test_tier_a_hash_labels_do_not_certify_original_publication(tmp_path):
    path,clocks=fixture(tmp_path)
    clocks[0].update(pit_tier='A',original_vintage_hash='a'*64,public_evidence_hash='b'*64)
    with pytest.raises(ValueError,match='verified original/public evidence'):parse_nport(path,clocks)
