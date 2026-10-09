"""FP64 share/cash ledger. Actions precede ex-date purchases; cash dividends held."""
from dataclasses import dataclass, field
import math


@dataclass
class Ledger:
    cash: float
    shares: dict = field(default_factory=dict)
    receivables: dict = field(default_factory=dict)
    costs: float = 0.0
    action_ids: set = field(default_factory=set)

    def __post_init__(self):
        self.cash=float(self.cash);self.costs=float(self.costs)
        self.shares={k:float(v) for k,v in self.shares.items()}
        self.receivables={k:float(v) for k,v in self.receivables.items()}
        if any(not math.isfinite(v) or v<0 for v in [self.cash,self.costs,*self.shares.values(),*self.receivables.values()]):
            raise ValueError('Finite nonnegative long-only ledger state required')

    def nav(self, prices):
        if any(not math.isfinite(float(p)) or p<=0 for p in prices.values()):
            raise ValueError('Missing/stale/nonpositive mark requires explicit nontradable handling')
        return self.cash + sum(v*prices[k] for k,v in self.shares.items()) + sum(self.receivables.values())

    def split(self, asset, ratio, action_id):
        if action_id in self.action_ids or not math.isfinite(ratio) or ratio<=0:
            raise ValueError('Duplicate or invalid action')
        self.shares[asset] = self.shares.get(asset,0)*ratio
        self.action_ids.add(action_id)

    def ex_dividend(self, asset, amount, action_id):
        if action_id in self.action_ids or not math.isfinite(amount) or amount<0:
            raise ValueError('Duplicate or invalid dividend')
        self.receivables[action_id] = self.shares.get(asset,0)*amount
        self.action_ids.add(action_id)

    def pay_dividend(self, action_id):
        self.cash += self.receivables.pop(action_id)

    def accrue_cash(self, asof_rate, days, daycount=365):
        if not math.isfinite(asof_rate) or days<0 or daycount<=0:
            raise ValueError('Invalid cash accrual')
        multiplier=1+asof_rate*days/daycount
        if multiplier<0:raise ValueError('Cash accrual would violate nonnegative long-only state')
        self.cash *= multiplier

    def rebalance(self, weights, prices, cost_bps=0):
        if (not math.isfinite(cost_bps) or cost_bps<0 or any(not math.isfinite(w) or w<0 for w in weights.values())
            or sum(weights.values())>1+1e-12):
            raise ValueError('Invalid long-only allocation')
        before = self.nav(prices)
        assets = sorted(set(self.shares)|set(weights))
        pretrade = {k:self.shares.get(k,0)*prices[k] for k in assets}
        # Solve the post-cost target NAV fixed point, avoiding negative cash at full exposure.
        nav = before
        for _ in range(100):
            target = {k:weights.get(k,0)*nav for k in assets}
            traded = sum(abs(target[k]-pretrade[k]) for k in assets)
            updated = before-traded*cost_bps/10000
            if abs(updated-nav)<=1e-12*max(1,before):
                nav=updated
                break
            nav=updated
        else:
            raise ValueError('Cost target fixed point failed')
        target = {k:weights.get(k,0)*nav for k in assets}
        traded = sum(abs(target[k]-pretrade[k]) for k in assets)
        fee = traded*cost_bps/10000
        new_cash = self.cash-sum(target[k]-pretrade[k] for k in assets)-fee
        if new_cash < -1e-8*max(1,before):
            raise ValueError('Self-financing cash infeasible')
        self.cash=new_cash
        self.shares={k:target[k]/prices[k] for k in assets}
        self.costs+=fee
        if abs(self.nav(prices)-(before-fee))>1e-8*max(1,before):
            raise ValueError('Accounting conservation failure')
        return {'traded_notional':traded,'fee':fee,'nav_before':before,'nav_after':self.nav(prices)}


def feasible_weights(means,cash_return,covariance,commodity_indices=(),max_asset=.25,max_commodity=.35,horizon_weeks=1,investable_fraction=1.):
    import numpy as np
    import cvxpy as cp
    mu,cov=np.asarray(means,dtype=float)-cash_return,np.asarray(covariance,dtype=float)
    if horizon_weeks not in {1,4} or not math.isfinite(investable_fraction) or not 0<=investable_fraction<=1:
        raise ValueError('Registered horizon and investable NAV fraction required')
    annualizer=52/horizon_weeks
    if cov.shape != (len(mu),len(mu)) or not np.isfinite(mu).all() or not np.isfinite(cov).all():
        raise ValueError('Invalid controller inputs')
    cov=(cov+cov.T)/2
    eig,vec=np.linalg.eigh(cov)
    cov=(vec*np.maximum(eig,1e-10))@vec.T
    w=cp.Variable(len(mu))
    constraints=[w>=0,w<=max_asset,cp.sum(w)<=investable_fraction,cp.quad_form(w,cp.psd_wrap(cov))*annualizer<=.1**2]
    if commodity_indices:
        constraints.append(cp.sum(w[list(commodity_indices)])<=max_commodity)
    problem=cp.Problem(cp.Maximize(mu@w-5*cp.quad_form(w,cp.psd_wrap(cov))),constraints)
    try:
        problem.solve(solver='CLARABEL')
    except cp.error.SolverError:
        return np.zeros(len(mu)), 'CASH_FALLBACK'
    if w.value is None or problem.status not in {'optimal','optimal_inaccurate'}:
        return np.zeros(len(mu)), 'CASH_FALLBACK'
    result=np.maximum(w.value,0)
    if (result.max()>max_asset+1e-7 or result.sum()>investable_fraction+1e-7 or
        (commodity_indices and result[list(commodity_indices)].sum()>max_commodity+1e-7) or result@cov@result*annualizer>.1**2+1e-7):
        return np.zeros(len(mu)), 'CASH_FALLBACK'
    return result,'QUALIFIED_SOLUTION'


def controller_horizon(forecast,risk,cash):
    """Mean, covariance and cash returns must use the same registered interval."""
    keys=['decision_time','horizon_weeks']
    for key in keys:
        if key not in forecast or risk.get(key)!=forecast[key] or cash.get(key)!=forecast[key]:
            raise ValueError('Controller horizon/decision mismatch: '+key)
    import pandas as pd
    decision=pd.Timestamp(forecast['decision_time'])
    if decision.tzinfo is None:raise ValueError('Aware controller decision required')
    for state in [risk,cash]:
        known=pd.Timestamp(state['max_dependency_available_at'])
        if known.tzinfo is None or known>decision:raise ValueError('Controller future dependency')
    return feasible_weights(forecast['mean'],cash['return'],risk['covariance'],horizon_weeks=forecast['horizon_weeks'])
