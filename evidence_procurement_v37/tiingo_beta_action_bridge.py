"""Tiingo-shaped external event reconciliation; originals remain immutable."""
import math
from evidence_procurement_v37.core import day

def reconcile(records,legacy,issuer,snapshot_asof):
    snapshot=day(snapshot_asof);out=[]
    for r in records:
        value=r['values'];kind=r['kind'];item={'event_id':r['event_id'],'version_id':r['version_id'],'kind':kind,'legacy_matches':0,'issuer_matches':0,'conflicts':[],'qualified_Tier_A':False,'P1_qualified':False,'test_not_evidence':r['test_not_evidence']}
        if kind=='dividend':
            asset=value['ticker'];ex=day(value['exDate']);amount=value['distribution'];cancel=value['distributionFrequency']=='c';pay=day(value['paymentDate']) if value['paymentDate'] else None
            old=[x for x in legacy if x['asset']==asset and x['ex_date']==ex and math.isclose(float(x['divCash']),float(amount),rel_tol=0,abs_tol=1e-10)];official=[x for x in issuer if x['asset']==asset and x['ex_date']==ex and math.isclose(float(x['amount']),float(amount),rel_tol=0,abs_tol=1e-10)]
            item.update(asset=asset,permaTicker=value['permaTicker'],ex_date=ex,amount=amount,payment_date=pay,declaration_date=day(value['declarationDate']) if value['declarationDate'] else None,legacy_matches=len(old),issuer_matches=len(official),state='CANCELLED_RETAINED_NOT_PAYMENT' if cancel else 'UNIQUE_EX_AMOUNT_MATCH_PARTIAL' if len(old)==1 else 'UNMATCHED_OR_AMBIGUOUS',payment_status='CANCELLED' if cancel else 'PAY_DATE_MISSING' if pay is None else 'SCHEDULED_AFTER_SNAPSHOT' if pay>snapshot else 'DATE_REACHED_NOT_PAYMENT_TRANSFER_PROOF',original_known_at=r['known_at'],legacy_mutated=False)
            if len(old)>1 or len(official)>1:item['conflicts'].append('ambiguous_duplicate_reference')
            for x in official:
                if pay and pay!=x['pay_date']:item['conflicts'].append('issuer_payment_date_difference')
        elif kind=='split':
            item.update(asset=value['ticker'],ex_date=day(value['exDate']),splitFrom=value['splitFrom'],splitTo=value['splitTo'],splitFactor=value['splitFactor'],state='ACTIVE_RATIO_CONTENT_ONLY' if value['splitStatus']=='a' else 'CANCELLED_RETAINED_NO_SHARE_CHANGE',share_change_allowed=False)
        elif kind=='no_action':item.update(state='NO_ACTION_CONTENT_REQUIRES_COMPLETENESS_AND_CLOCK_PROOF',state_is_not_inferred=True)
        else:item['state']='INDEPENDENT_EVIDENCE_LANE_NOT_ACTION_JOIN'
        out.append(item)
    return {'rows':out,'matched_legacy_rows':sum(x['legacy_matches']==1 and x.get('state')=='UNIQUE_EX_AMOUNT_MATCH_PARTIAL' for x in out),'issuer_payment_conflicts':sum('issuer_payment_date_difference' in x['conflicts'] for x in out),'new_real_provider_matches':0 if all(x['test_not_evidence'] for x in records) else sum(x['legacy_matches']==1 for x in out),'legacy_rows_mutated':0,'P1_qualified_sessions':0,'Tier_A_qualified':0}
