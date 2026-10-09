"""Compare partial issuer evidence without rewriting provider action receipts."""
import json
from pathlib import Path
import pandas as pd
from signalforge.runtime import atomic_json,file_hash
R=Path('/mnt/c/Users/USERNAME/Downloads/SignalForge-QX-v33-dev');O=Path('/mnt/c/Users/USERNAME/Downloads/SignalForge-QX');T=Path('/home/USERNAME/.local/share/signalforge-qx-v3/da768f7446b1')
def main():
    p=T/'artifacts/tiingo_inputs/ba9be84158dedc267ba5191b298084ffa5cdc4e7dbe92bb105e828f7dd4062f0/p1_preparation.json';d=json.loads(p.read_text());issuer=json.loads((O/'reports/issuer_action_evidence_2023.json').read_text());actions=pd.DataFrame(d['actions']);rows=[]
    for i in issuer['rows']:
        ex=pd.Timestamp(i['ex_date']).date().isoformat();matched=actions[(actions.asset==i['asset'])&(actions.ex_date==ex)];row={**i,'normalized_ex_date':ex,'provider_matches':len(matched),'pay_date_authentic_issuer_retrospective':True,'first_announcement_clock_qualified':False,'P1_qualified':False}
        if len(matched)==1:
            row['provider_divCash']=float(matched.iloc[0].divCash);row['distribution_difference']=row['distribution_per_share']-row['provider_divCash'];row['provider_missing_pay_date']=bool(pd.isna(matched.iloc[0].pay_date))
        rows.append(row)
    out={'issuer_pdf_sha256':issuer['pdf_sha256'],'provider_receipt_sha256':file_hash(p),'issuer_rows':len(rows),'unique_provider_matches':sum(r['provider_matches']==1 for r in rows),'provider_action_rows':len(actions),'provider_missing_pay_dates':int(actions.pay_date.isna().sum()),'matched_issuer_pay_dates':sum(r['provider_matches']==1 for r in rows),'no_complete_coverage_claim':True,'original_announcement_clocks_qualified':False,'rows':rows,'reserved_access':False};atomic_json(R/'reports/p1_pit_v34/issuer_action_reconciliation.json',out);pd.DataFrame(rows).to_csv(R/'reports/p1_pit_v34/issuer_action_reconciliation.csv',index=False);print(json.dumps({k:v for k,v in out.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()