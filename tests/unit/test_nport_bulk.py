import json,zipfile
from pathlib import Path
import pandas as pd
import pytest
from signalforge.nport_bulk import load_plan,target_accessions,target_holding_summaries,parse_accepted,canonical_rows,rows_to_events,holdings_to_events,accession_index_url
from signalforge.sources import validate_payload

REPO=Path(__file__).parents[2]


def fixture_zip(path):
    access=['0000000001-23-000001','0000000002-23-000002','0000000003-23-000003','0000000004-23-000004','0000000099-23-000099']
    submission=pd.DataFrame({'ACCESSION_NUMBER':access,'FILING_DATE':['2023-05-30']*5,'FILE_NUM':['811-x']*5,'SUB_TYPE':['NPORT-P']*5,
        'REPORT_ENDING_PERIOD':['2023-09-30']*5,'REPORT_DATE':['2023-03-31']*5,'IS_LAST_FILING':['N']*5})
    reg=pd.DataFrame({'ACCESSION_NUMBER':access,'CIK':['0000884394','0001067839','0001100663','0001100663','0000000099'],
        'REGISTRANT_NAME':['SPDR S&P 500 ETF TRUST','INVESCO QQQ TRUST, SERIES 1','iSHARES TRUST','iSHARES TRUST','OTHER FUND']})
    def row(acc,name,sid,base):
        d={'ACCESSION_NUMBER':acc,'SERIES_NAME':name,'SERIES_ID':sid,'NET_ASSETS':str(base*100)}
        for i in range(1,4):d.update({f'SALES_FLOW_MON{i}':str(base+i),f'REDEMPTION_FLOW_MON{i}':str(i),f'REINVESTMENT_FLOW_MON{i}':str(i/10)})
        return d
    fund=pd.DataFrame([row(access[0],None,None,100),row(access[1],'Invesco QQQ Trust, Series 1','S_HIST_QQQ',200),
        row(access[2],'iShares 7-10 Year Treasury Bond ETF','S000004358',300),row(access[3],'iShares 20+ Year Treasury Bond ETF','S000004360',400),
        row(access[4],'Other','S_OTHER',500)])
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,frame in [('SUBMISSION.tsv',submission),('REGISTRANT.tsv',reg),('FUND_REPORTED_INFO.tsv',fund)]:z.writestr(name,frame.to_csv(sep='\t',index=False))
        holding=pd.DataFrame([
            {'ACCESSION_NUMBER':access[0],'HOLDING_ID':'1','PERCENTAGE':'60'},
            {'ACCESSION_NUMBER':access[0],'HOLDING_ID':'2','PERCENTAGE':'40'},
            {'ACCESSION_NUMBER':access[1],'HOLDING_ID':'3','PERCENTAGE':'55'},
            {'ACCESSION_NUMBER':access[1],'HOLDING_ID':'4','PERCENTAGE':'45'},
            {'ACCESSION_NUMBER':access[4],'HOLDING_ID':'5','PERCENTAGE':'100'}])
        z.writestr('FUND_REPORTED_HOLDING.tsv',holding.to_csv(sep='\t',index=False))


def test_plan_applicability_and_bulk_target_discovery(tmp_path):
    plan,_=load_plan(REPO);assert set(plan['targets'])=={'SPY','QQQ','IEF','TLT'}
    assert sorted(plan['non_applicable_assets'])==['GLD','SLV','UNG','USO']
    path=tmp_path/'nport.zip';fixture_zip(path);validation=validate_payload(path,'nport_zip')
    assert validation['target_tables_only'] and len(validation['crc_verified_members'])==3
    selected=target_accessions(path,plan,'2023Q2')
    assert set(selected.asset)=={'SPY','QQQ','IEF','TLT'} and len(selected)==4
    assert selected.loc[selected.asset.eq('SPY'),'SERIES_ID'].isna().all()
    assert selected.loc[selected.asset.eq('QQQ'),'SERIES_ID'].iloc[0]=='S_HIST_QQQ'
    holdings=target_holding_summaries(path,selected)
    spy=holdings[holdings.asset.eq('SPY')].iloc[0];qqq=holdings[holdings.asset.eq('QQQ')].iloc[0]
    assert spy.holding_count==2 and spy.top10_abs_percentage==100 and spy.gross_abs_percentage==100
    assert qqq.holding_count==2 and qqq.top10_abs_percentage==100


def test_acceptance_clock_and_reference_month_flows(tmp_path):
    plan,_=load_plan(REPO);path=tmp_path/'nport.zip';fixture_zip(path);selected=target_accessions(path,plan,'2023Q2')
    accepted=parse_accepted('<div>Accepted</div><div>2023-05-30 16:12:03</div>')
    assert accepted==pd.Timestamp('2023-05-30 16:12:03',tz='America/New_York').tz_convert('UTC')
    clocks=[{'ACCESSION_NUMBER':a,'available_at':(accepted+pd.Timedelta(minutes=15)).isoformat(),'clock_evidence':'https://www.sec.gov/x','clock_evidence_hash':'a'*64} for a in selected.ACCESSION_NUMBER]
    rows,blocked=canonical_rows(selected,clocks,'b'*64)
    assert not blocked and len(rows)==12 and set(rows.reference_month)=={'2023-01','2023-02','2023-03'}
    spy=rows[rows.asset.eq('SPY')].sort_values('reference_month').iloc[0]
    assert spy.external_flow==100 and spy.reinvestment==.1 and spy.reported_end_net_assets==10000
    events=rows_to_events(rows);assert len(events)==60 and events.pit_tier.eq('B').all() and not events.original_publication_qualified.any()


def test_sec_accession_url_is_exact():
    assert accession_index_url('0001100663','0002071691-26-008193')=='https://www.sec.gov/Archives/edgar/data/1100663/000207169126008193/0002071691-26-008193-index.htm'
    with pytest.raises(ValueError):accession_index_url('1100663','bad')
