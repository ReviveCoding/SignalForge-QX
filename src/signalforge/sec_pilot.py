"""Existing official 2022Q1 pilot; independent timing proof cannot be invented."""
import json,os
import pandas as pd
from .runtime import atomic_json
from .sources import Acquisition,INDEX,discover_zip,validate_sec_user_agent


def acquire_nport_pilot(repo,runtime):
    validate_sec_user_agent(os.environ.get('SEC_USER_AGENT',''))
    # SEC's official 2022Q1 bulk file is ~460 MB compressed. Use the dedicated
    # selective N-PORT validator under a bounded 512 MiB transport budget.
    client=Acquisition(runtime,512*1024**2)
    html,index=client.html(INDEX['nport']);url=discover_zip(html,'nport','2022Q1')
    raw=client.cached(url,'nport_zip') or client.fetch(url,'nport_zip',metadata={'quarter':'2022Q1','reserved_access':False})
    result={'source':'nport','state':'SUCCEEDED_RAW_DISSEMINATION_BLOCKED','raw':raw,'quarter':'2022Q1',
            'reserved_access':False,'model_qualified':False,'qualified_for_final':False,
            'dissemination':{'state':'BLOCKED_DATA','reason':'Hash-bound historical official dissemination evidence absent; filing date is not public availability'}}
    cardpath=repo/'.local/nport_dissemination_card.json'
    if cardpath.exists():
        from .track_inputs import read_input_card
        from .nport import parse_nport
        clocks=read_input_card(runtime,json.loads(cardpath.read_text()))
        for clock in clocks:
            t=pd.Timestamp(clock['available_at'])
            if t.tzinfo is None or t>=pd.Timestamp('2024-01-01T00:00Z'):raise PermissionError('Development-only dissemination evidence required')
            if clock['pit_tier']!='B':raise PermissionError('This reconstruction path does not qualify Tier A')
        flows,missing=parse_nport(raw['path'],clocks)
        if len(flows) and pd.to_datetime(flows.report_date).ge('2024-01-01').any():raise PermissionError('Reserved NPORT reports forbidden')
        result['dissemination']={'state':'RECONSTRUCTED_TIER_B' if not missing else 'PARTIAL_BLOCKED_DATA',
            'flows':json.loads(flows.to_json(orient='records')),'unqualified':missing,'qualified_for_final':False}
    atomic_json(repo/'reports/nport_development_acquisition.json',result)
    return result
