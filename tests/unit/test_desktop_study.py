import json
from pathlib import Path
import pytest
from signalforge.desktop_study import validate_study


def documents():
    root=Path(__file__).resolve().parents[2]/'research/desktop_study'
    return [json.loads((root/n).read_text()) for n in ['question_contract.json','literature_matrix.json','data_dictionary.json','source_access_matrix.json','mechanism_cards.json','method_source_decisions.json']]


def test_track_targets_and_users_cannot_be_silently_collapsed():
    docs=documents();assert validate_study(*docs)['data_or_PIT_certification'] is False
    docs[0]['tracks'].pop('Nested-B')
    with pytest.raises(ValueError,match='three distinct'):validate_study(*docs)
    docs=documents();docs[0]['tracks']['Main-A'].pop('user')
    with pytest.raises(ValueError,match='track user'):validate_study(*docs)


def test_prior_method_claim_requires_reference_and_implementation_difference():
    docs=documents();docs[1][0].pop('implementation_difference')
    with pytest.raises(ValueError,match='attribution'):validate_study(*docs)
    docs=documents();docs[1]=[r for r in docs[1] if r['method']!='TFT']
    with pytest.raises(ValueError,match='lacks its attribution'):validate_study(*docs)


def test_positioning_cannot_be_labeled_cash_flow():
    docs=documents();docs[0]['positioning_is_cash_flow']=True
    with pytest.raises(ValueError,match='CFTC'):validate_study(*docs)
    docs=documents();docs[2][0]['unit']='USD cash flow'
    with pytest.raises(ValueError,match='Position units'):validate_study(*docs)


def test_access_policy_requires_every_source_license_auth_and_redistribution():
    for key in ['license','auth','rate_limit','redistribution','size']:
        docs=documents();docs[3][0].pop(key)
        with pytest.raises(ValueError,match='source contract'):validate_study(*docs)
    docs=documents();docs[3].pop()
    with pytest.raises(ValueError,match='Source definition'):validate_study(*docs)


def test_mechanism_requires_competitor_and_falsification():
    for key in ['alternative','falsification']:
        docs=documents();docs[4][0].pop(key)
        with pytest.raises(ValueError,match='competing explanation'):validate_study(*docs)


def test_dictionary_cannot_omit_units_entity_or_public_reference_clock():
    for key in ['unit','entity','public_clock','reference_clock']:
        docs=documents();docs[2][0].pop(key)
        with pytest.raises(ValueError,match='source contract'):validate_study(*docs)


def test_established_components_cannot_be_claimed_as_standalone_novelty():
    docs=documents();docs[0]['standalone_novelty_methods']=['TFT']
    with pytest.raises(ValueError,match='standalone novelty'):validate_study(*docs)


def test_method_source_change_requires_registered_reason_and_adr():
    docs=documents();docs[5][0].pop('adr')
    with pytest.raises(ValueError,match='ADR'):validate_study(*docs)
    docs=documents();docs[5][0]['decision']='silently_replace'
    with pytest.raises(ValueError,match='Unregistered'):validate_study(*docs)
