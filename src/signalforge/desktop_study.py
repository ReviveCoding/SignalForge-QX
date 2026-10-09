"""Validate research definitions and attribution without certifying source data."""
from datetime import date
from urllib.parse import urlparse


def validate_study(contract, literature, dictionary, access, mechanisms, decisions):
    tracks=contract['tracks']
    if set(tracks)!={'Main-A','Nested-B','Auxiliary-C'}:
        raise ValueError('All three distinct track definitions required')
    for definition in tracks.values():
        for field in ['user','target','decision','claim_boundary']:
            if not isinstance(definition.get(field),str) or not definition[field].strip():
                raise ValueError('Missing track '+field)
    if tracks['Auxiliary-C'].get('execution_grade') is not False:
        raise ValueError('Inventory predictions cannot establish execution-grade economics')
    if contract.get('positioning_is_cash_flow') is not False:
        raise ValueError('CFTC positioning is not reported dollar cash flow')
    if contract.get('nport_external_flow')!='reported_sales_minus_redemptions_reinvestment_separate':
        raise ValueError('N-PORT external-flow semantics changed')
    established=set(contract['established_methods'])
    if not {'GRU-D','TFT'}.issubset(established) or contract.get('standalone_novelty_methods'):
        raise ValueError('Established components cannot be standalone novelty')
    methods=[]
    for row in literature:
        methods.append(row['method'])
        for key in ['question','baseline','implementation_difference','limitation','validated_read_scope']:
            if not row.get(key):raise ValueError('Missing literature attribution '+key)
        if urlparse(row.get('url','')).scheme!='https':raise ValueError('Primary reference required')
    if len(methods)!=len(set(methods)) or not established.issubset(methods):
        raise ValueError('Prior-method claim lacks its attribution record')
    required={'cftc','nport','fred','eia','market'}
    for rows,fields in [(dictionary,['unit','entity','reference_clock','public_clock','revision','url','access_date']),
                        (access,['auth','license','rate_limit','size','url','access_date','redistribution'])]:
        if len(rows)!=len(required) or {r['id'] for r in rows}!=required:
            raise ValueError('Source definition missing or duplicated')
        for row in rows:
            if any(not row.get(k) for k in fields):raise ValueError('Incomplete source contract')
            date.fromisoformat(row['access_date'])
            if urlparse(row['url']).scheme!='https':raise ValueError('Official source URL required')
    by_id={r['id']:r for r in dictionary}
    if by_id['cftc']['unit']!='contracts':raise ValueError('Position units cannot become cash flow')
    for card in mechanisms:
        if any(not card.get(k) for k in ['id','hypothesis','alternative','falsification','features','claim_boundary']):
            raise ValueError('Mechanism requires competing explanation and falsification')
    if len({r['id'] for r in mechanisms})!=len(mechanisms) or not mechanisms:
        raise ValueError('Mechanism identities missing or duplicated')
    for row in decisions:
        if row.get('decision') not in {'retain','defer','omit','substitute_transport','compact_variant'}:
            raise ValueError('Unregistered method/source change')
        if any(not row.get(k) for k in ['item','reason','adr','claim_boundary']):
            raise ValueError('Every source/method change needs an ADR and claim boundary')
    return {'state':'VALIDATED_DESKTOP_CONTRACTS','tracks':sorted(tracks),'source_count':len(required),
            'literature_records':len(literature),'mechanism_count':len(mechanisms),
            'data_or_PIT_certification':False,'empirical_novelty_established':False}
