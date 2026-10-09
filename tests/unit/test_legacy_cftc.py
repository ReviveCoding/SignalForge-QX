import zipfile
import pytest
from signalforge.sources import parse_cftc


def test_legacy_cftc_date_schema_agreement(tmp_path):
    archive=tmp_path/'legacy.zip'
    header='Market_and_Exchange_Names,Report_Date_as_MM_DD_YYYY,As_of_Date_In_Form_YYMMDD,CFTC_Contract_Market_Code,Open_Interest_All\n'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('FinFutYY.txt',header+'TEST,2010-12-28,101228,090741,113223\n')
    frame=parse_cftc(archive,'tff')
    assert str(frame.report_date.iloc[0].date())=='2010-12-28'
    assert frame.CFTC_Contract_Market_Code.iloc[0]=='090741' and frame.Market_and_Exchange_Names.iloc[0]=='TEST'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('FinFutYY.txt',header+'TEST,2010-12-28,101227,090741,113223\n')
    with pytest.raises(ValueError,match='date fields disagree'):parse_cftc(archive,'tff')
