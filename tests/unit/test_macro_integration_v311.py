import math
from signalforge.macro_integration import _indpro_yoy,_stable_level_event

def row(reference,vintage,value,raw):
    return {'series_id':'INDPRO','reference_time':reference+'T00:00:00+00:00','realtime_start':vintage,
            'available_at':vintage+'T12:00:00+00:00','value':value,'raw_hash':raw*64}

def test_indpro_same_vintage_yoy_recomputes_when_denominator_revised():
    rows=[
        row('2010-01-31','2011-02-01',100.0,'a'),
        row('2011-01-31','2011-02-01',110.0,'b'),
        row('2010-01-31','2011-03-01',105.0,'c'),
    ]
    events=_indpro_yoy(rows)
    target=[e for e in events if e['reference_time'].startswith('2011-01-31')]
    assert len(target)==2
    first=next(e for e in target if e['fred_vintage_date']=='2011-02-01')
    revised=next(e for e in target if e['fred_vintage_date']=='2011-03-01')
    assert math.isclose(first['value'],10.0)
    assert math.isclose(revised['value'],100*(110/105-1))
    assert revised['unit']=='percent_yoy'
    assert revised['source']=='macro'
    assert len(revised['raw_hash'])==64

def test_stable_macro_level_preserves_registered_units():
    event=_stable_level_event({'series_id':'DGS10','unit':'Percent','reference_time':'2020-01-02T00:00:00+00:00',
        'available_at':'2020-01-03T12:00:00+00:00','value':1.88,'raw_hash':'d'*64,
        'realtime_start':'2020-01-02','clock_evidence':'date_only'})
    assert event['unit']=='Percent' and event['field']=='value' and event['entity']=='DGS10'
    assert event['original_publication_qualified'] is False
