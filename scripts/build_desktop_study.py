"""Write source-linked desktop artifacts from explicit project assumptions."""
from signalforge.runtime import paths,atomic_json

repo,_=paths();folder=repo/'research/desktop_study'
required=['question_contract.json','literature_matrix.json','data_dictionary.json',
          'source_access_matrix.json','mechanism_cards.json','method_source_decisions.json']
if all((folder/n).is_file() for n in required):
    # Curated, verified definitions are inputs, not disposable scaffolding.
    # A resumed pipeline validates them instead of overwriting their history.
    import os,sys
    os.execv(sys.executable,[sys.executable,str(repo/'scripts/audit_desktop_study.py')])


def seed(path,value):
    if not path.exists():atomic_json(path,value)

access=[
 {'id':'cftc','url':'https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm',
  'auth':'none','unit':'contracts','entity':'CFTC contract code','reference_clock':'report Tuesday/date field',
  'public_clock':'actual release evidence required; nominal lag insufficient','revision':'annual archive original vintage unverified',
  'license':'public official archive; no raw public redistribution performed','rate_limit':'project 1 request/sec; 403/429 stop',
  'size':'2022 pilot compressed TFF 494559 bytes, disaggregated 2049193 bytes','access_date':'2026-10-02'},
 {'id':'nport','url':'https://www.sec.gov/data-research/sec-markets-data/form-n-port-data-sets',
  'auth':'genuine SEC_USER_AGENT missing','unit':'reported currency amounts with currency metadata','entity':'accession/series/reference month',
  'reference_clock':'actual report month; MON1–3 aligned from report date','public_clock':'public dissemination evidence, amendments retained',
  'revision':'accessions/amendments must be separate','license':'SEC public dataset; documented errors/metadata limits; redistribution review pending',
  'rate_limit':'project <=1 request/sec; never bypass denial','size':'quarter pilot pending genuine contact','access_date':'2026-10-02'},
 {'id':'fred','url':'https://fred.stlouisfed.org/docs/api/fred/series_observations.html',
  'auth':'FRED_API_KEY missing','unit':'series-specific, never treat yields as returns','entity':'series/reference date/realtime interval',
  'reference_clock':'observation date','public_clock':'conservative realtime date bound with release evidence',
  'revision':'initial snapshot plus new/revised updates and pagination','license':'series source restrictions require review; no redistribution',
  'rate_limit':'project <=1 request/sec; entitlement denial blocks','size':'bounded paginated JSON; unmeasured','access_date':'2026-10-02'},
 {'id':'eia','url':'https://www.eia.gov/petroleum/supply/weekly/archive/',
  'auth':'archive CSV no key','unit':'table4 million barrels; table1 mixed stocks/rates; table9 mixed rates/percent',
  'entity':'series/PADD/issue/reference date','reference_clock':'CSV dated columns',
  'public_clock':'archive issue date conservative upper bound; exceptional timing retained',
  'revision':'each issue immutable; archive original authentication unverified',
  'license':'official public statistics; no raw external redistribution performed',
  'rate_limit':'project <=1 request/sec, bounded single worker','size':'2022 pilot tables1/4/9 6092/2355/45779 bytes',
  'access_date':'2026-10-02'},
 {'id':'market','url':'https://www.alphavantage.co/documentation/',
  'auth':'ALPHAVANTAGE_API_KEY missing; daily-adjusted documented premium; no purchase',
  'unit':'currency/share plus raw actions','entity':'symbol/session/valid mapping','reference_clock':'exchange session',
  'public_clock':'provider data availability/corrections must qualify','revision':'raw vs adjusted/action convention required',
  'license':'provider entitlement/license unqualified; no alternative substituted',
  'rate_limit':'provider entitlement controls; no calls without existing key','size':'weekly/full daily unmeasured','access_date':'2026-10-02'}]
for row in access:row['redistribution']='No external raw redistribution performed; source and series-specific rights require review before distribution'
seed(folder/'source_access_matrix.json',access)
seed(folder/'data_dictionary.json',[{k:v for k,v in row.items() if k in {'id','unit','entity','reference_clock','public_clock','revision','url','access_date'}} for row in access])
literature=[
 {'method':'GRU-D','url':'https://arxiv.org/abs/1606.01865','question':'ragged missing multivariate inputs',
  'validated_read_scope':'primary abstract and bibliographic identity; full algorithm reproduction not asserted',
  'implementation_difference':'compact input decay toward train-centered zero and recurrent encoder; not full hidden-state decay formulation',
  'baseline':'GRU with identical value/mask/age inputs','limitation':'paper healthcare evidence is not financial transport evidence'},
 {'method':'TFT','url':'https://arxiv.org/abs/1912.09363','question':'gated temporal quantile prediction',
  'validated_read_scope':'primary abstract and bibliographic identity; exact reproduction not asserted',
  'implementation_difference':'compact feature gate, GRU, causal attention, output gate; no full static enrichment or multi-horizon decoder',
  'baseline':'same-information GRU/concat transformer','limitation':'paper performance is not this study performance'},
 {'method':'LightGBM CUDA','url':'https://lightgbm.readthedocs.io/en/v4.6.0/Installation-Guide.html',
  'question':'strong nonlinear baseline','validated_read_scope':'official versioned installation/backend guide',
  'implementation_difference':'CUDA regression and independent quantile objectives; common registered rearrangement',
  'baseline':'all-information age/mask/coverage aware','limitation':'build/backend correctness is distinct from predictive value'},
 {'method':'XGBoost CUDA','url':'https://xgboost.readthedocs.io/en/release_3.0.0/gpu/index.html',
  'question':'second strong nonlinear comparator','validated_read_scope':'official versioned GPU guide',
  'implementation_difference':'hist + cuda:0, separate mean and multi-quantile outputs','baseline':'same-grid LightGBM',
  'limitation':'saved actual device inspected; CPU prediction/preprocessing share must be reported'}]
seed(folder/'literature_matrix.json',literature)
mechanisms=[
 {'id':'crowding','hypothesis':'persistent crowding/deceleration relates to downside distribution',
  'alternative':'positions chase price momentum; classification or release lag artifact',
  'features':['net/OI','lagged change','volatility interaction'],
  'falsification':'effect disappears with price controls or appears only in reconstructed vintage data',
  'claim_boundary':'conditional predictive relationship, not causal investor pressure'},
 {'id':'cohort','hypothesis':'aligned reported external flows have incremental information',
  'alternative':'entry/exit, mergers, reference-month mixing or concentration',
  'features':['sales','redemptions','reinvestment separate','fixed-month cohort coverage'],
  'falsification':'matched availability-known cohort removes apparent effect',
  'claim_boundary':'reported measure, not exact organic flow or secondary-market order flow'},
 {'id':'physical','hypothesis':'past stocks and release ages inform next initial reported inventory change',
  'alternative':'seasonality, revision/rebenchmarking and stale release selection',
  'features':['lagged commercial stocks changes','gasoline/distillate changes','release/reference ages'],
  'falsification':'does not improve seasonal/historical/ridge/tree baseline on chronological development folds',
  'claim_boundary':'inventory forecast, not tradable crude futures P&L'}]
seed(folder/'mechanism_cards.json',mechanisms)
print('Desktop artifact files written; methodological reproduction and source qualification remain separate.')
