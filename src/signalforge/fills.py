"""FP64 fill reconciliation: embedded price impact is not charged twice."""
import math


def execute_fills(ledger,fills,mid_prices):
    """Apply raw fills atomically; Ledger.costs tracks explicit cash fees only.

    Signed embedded amounts must reconcile to the actual fill-minus-mark impact.
    Price improvement remains a negative embedded cost, rather than being erased.
    This accounting oracle does not qualify a real price/execution data source.
    """
    before=ledger.nav(mid_prices);cash=float(ledger.cash);shares=dict(ledger.shares)
    explicit=0.;embedded=0.;records=[]
    if not fills:raise ValueError('Nonempty explicit fill batch required')
    for fill in fills:
        asset=fill['asset'];quantity=float(fill['quantity']);price=float(fill['price'])
        if (fill.get('adjusted') is not False or not math.isfinite(quantity) or quantity==0 or
            not math.isfinite(price) or price<=0 or asset not in mid_prices):
            raise ValueError('Finite signed quantity and explicitly raw positive fill price required')
        impact=quantity*(price-mid_prices[asset]);included=0.;fees=0.;kinds=set()
        components=fill.get('cost_components',[])
        for component in components:
            kind=component['kind'];amount=float(component['amount']);in_fill=component['included_in_fill']
            if (kind not in {'spread','slippage','commission','other_fee'} or kind in kinds or
                not isinstance(in_fill,bool) or not math.isfinite(amount) or
                (kind!='slippage' and amount<0) or (not in_fill and amount<0)):
                raise ValueError('Invalid or duplicated cost component; cannot charge embedded spread/slippage twice')
            kinds.add(kind)
            if in_fill:included+=amount
            else:fees+=amount
        if abs(included-impact)>1e-10*max(1,abs(quantity*price)):
            raise ValueError('Embedded cost components do not reconcile to actual fill price impact')
        cash-=quantity*price+fees;shares[asset]=shares.get(asset,0.)+quantity
        if shares[asset]<-1e-10:raise ValueError('Fill would introduce a short position')
        explicit+=fees;embedded+=impact
        records.append({'asset':asset,'quantity':quantity,'raw_fill_price':price,'mid_mark':mid_prices[asset],
            'embedded_execution_cost':impact,'additional_cash_fee':fees})
    if cash<-1e-10*max(1,before):raise ValueError('Self-financing fill batch cash infeasible')
    after=cash+sum(v*mid_prices[k] for k,v in shares.items())+sum(ledger.receivables.values())
    if abs(after-(before-embedded-explicit))>1e-10*max(1,before):raise ValueError('Fill gross-to-net conservation failure')
    ledger.cash=cash;ledger.shares=shares;ledger.costs+=explicit
    return {'nav_before':before,'nav_after':after,'embedded_execution_cost':embedded,'additional_cash_fees':explicit,
        'total_execution_cost':embedded+explicit,'cash_fee_book_excludes_embedded_fill_impact':True,
        'fills':records,'data_source_qualified':False}
