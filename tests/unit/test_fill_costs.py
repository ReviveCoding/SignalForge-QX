import copy
import pytest
from signalforge.portfolio import Ledger
from signalforge.fills import execute_fills


def test_embedded_spread_is_not_deducted_again_and_infeasible_batch_is_atomic():
    ledger=Ledger(2000.)
    fill={'asset':'A','quantity':10.,'price':101.,'adjusted':False,'cost_components':[
        {'kind':'spread','amount':10.,'included_in_fill':True},{'kind':'commission','amount':2.,'included_in_fill':False}]}
    result=execute_fills(ledger,[fill],{'A':100.})
    assert ledger.cash==988 and ledger.shares['A']==10 and ledger.costs==2
    assert result['nav_after']==1988 and result['total_execution_cost']==12
    before=copy.deepcopy(ledger)
    doubled=copy.deepcopy(fill);doubled['cost_components'].append({'kind':'spread','amount':10.,'included_in_fill':False})
    with pytest.raises(ValueError,match='twice'):execute_fills(ledger,[doubled],{'A':100.})
    assert ledger==before
    too_big=copy.deepcopy(fill);too_big['quantity']=100;too_big['cost_components'][0]['amount']=100
    with pytest.raises(ValueError,match='cash infeasible'):execute_fills(ledger,[too_big],{'A':100.})
    assert ledger==before
    raw_mid=Ledger(2000.);fill['price']=100;fill['cost_components'][0]['included_in_fill']=False
    assert execute_fills(raw_mid,[fill],{'A':100.})['nav_after']==1988


def test_real_price_improvement_is_preserved_in_embedded_slippage():
    ledger=Ledger(2000.)
    fill={'asset':'A','quantity':10.,'price':99.,'adjusted':False,
          'cost_components':[{'kind':'slippage','amount':-10.,'included_in_fill':True}]}
    result=execute_fills(ledger,[fill],{'A':100.})
    assert result['total_execution_cost']==-10 and ledger.nav({'A':100.})==2010 and ledger.costs==0
