"""Conservative Tier-B N-PORT development clock; never strict publication evidence."""
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

BUSINESS_DAY=CustomBusinessDay(calendar=USFederalHolidayCalendar())

def conservative_development_clock(value):
    """Return 23:59:59 ET on the next US-federal business day after EDGAR Accepted.

    SEC states filings are often available within 1-3 minutes of EDGAR acceptance,
    but does not guarantee that lag, and many submissions after 17:30 ET are not
    disseminated until the next business day. This deliberately delayed clock is
    development-only Tier B reconstruction, not original-publication proof.
    """
    t=pd.Timestamp(value)
    if t.tzinfo is None:
        raise ValueError('Aware EDGAR Accepted timestamp required')
    local=t.tz_convert('America/New_York')
    next_day=local.normalize().tz_localize(None)+BUSINESS_DAY
    return pd.Timestamp(next_day.date()).tz_localize('America/New_York')+pd.Timedelta(hours=23,minutes=59,seconds=59)
