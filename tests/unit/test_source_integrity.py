from copy import deepcopy
import pytest
from signalforge.data import auxiliary_rows
from signalforge.sources import parse_eia_stocks_pdf
from tests.unit.test_auxiliary_protocol import fixture_records


def test_known_raw_target_rejected():
    records=fixture_records();records[1]['raw']['sha256']=records[0]['raw']['sha256']
    with pytest.raises(ValueError,match='already-known'):auxiliary_rows(records)


def test_nonadvancing_reference_rejected():
    records=fixture_records();records[1]['event']['reference_time']=records[0]['event']['reference_time']
    with pytest.raises(ValueError,match='already-known'):auxiliary_rows(records)


def pdf_text():
    return '''Table 4. Stocks of Crude Oil (Million Barrels)
Current Week Last Week
6/28/19 6/21/19
Crude Oil .... 1,113.3 1,114.4 -1.1
Commercial (Excluding SPR)3 .... 468.5 469.6 -1.1
Total Motor Gasoline7 .... 230.6 232.2 -1.6
Distillate Fuel Oil7 .... 126.8 125.4 1.4
Cushing4 .... 52.5 51.8 0.7
SPR6 .... 644.8 644.8 0.0
'''


def test_published_pdf_precision_and_clock():
    event=parse_eia_stocks_pdf(pdf_text(),'https://www.eia.gov/petroleum/supply/weekly/archive/2019/2019_07_03/wpsr_2019_07_03.php')
    assert event['series']['Commercial (Excluding SPR)']['change']==-1.1
    assert event['reference_time']=='2019-06-28T00:00:00+00:00'
    assert event['published_precision']==.1 and event['pit_tier']=='B'


def test_pdf_wrong_unit_or_difference_rejected():
    url='https://www.eia.gov/petroleum/supply/weekly/archive/2019/2019_07_03/wpsr_2019_07_03.php'
    for bad in [pdf_text().replace('Million Barrels','Thousand Barrels'),pdf_text().replace('468.5','460.5')]:
        with pytest.raises(ValueError):parse_eia_stocks_pdf(bad,url)


def test_failed_trailing_source_retains_missing_origin_and_known_context():
    records=fixture_records();last=records[-1]
    last['release_page']='https://www.eia.gov/petroleum/supply/weekly/archive/2017/2017_10_04/wpsr_2017_10_04.php'
    last['state']='FAILED_SOURCE';last.pop('event')
    frame=auxiliary_rows(records)
    failed=frame[frame.target_status.eq('FAILED_SOURCE_TARGET')]
    unknown=frame[frame.target_status.eq('MISSING_NEXT_RELEASE_METADATA')]
    assert len(failed)==1 and len(unknown)==1
    assert failed.y.isna().all() and unknown.y.isna().all()
    assert unknown.label_available_at.isna().all()
    assert frame.x.map(len).eq(14).all()
