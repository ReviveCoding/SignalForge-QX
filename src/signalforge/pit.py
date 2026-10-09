"""Release-aware event selection. Availability timestamps must be supplied, not guessed."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _aware_utc(values, name: str) -> pd.Series:
    """Reject ambiguous naive timestamps rather than silently treating them as UTC."""
    seq = pd.Series(values)
    for value in seq:
        if pd.isna(value) or pd.Timestamp(value).tzinfo is None:
            raise ValueError(f"{name} requires non-null timezone-aware timestamps")
    return pd.to_datetime(seq, utc=True, format='mixed')


def asof_snapshot(events: pd.DataFrame, decision_time, strict: bool = False) -> pd.DataFrame:
    """Latest usable reference per entity/source/field, with latest known revision.

    Required: entity, source, field, reference_time, available_at, value.
    strict=True also requires pit_tier='A'. available_at is a conservative upper
    bound of actual public availability, NEVER just report date or download time.
    Ambiguous version ties are rejected. Ingestion time remains separate metadata.
    """
    keys = ["entity", "source", "field"]
    required = set(keys + ["reference_time", "available_at", "value"])
    if not required <= set(events.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(events.columns))}")
    t = pd.Timestamp(decision_time)
    if t.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    df = events.copy()
    if df[keys].isna().any().any():
        raise ValueError("Null event keys are not permitted")
    df["reference_time"] = _aware_utc(df["reference_time"], "reference_time").to_numpy()
    df["available_at"] = _aware_utc(df["available_at"], "available_at").to_numpy()
    # Explicitly restore pandas UTC dtypes after positional assignment.
    df["reference_time"] = pd.to_datetime(df["reference_time"], utc=True, format='mixed')
    df["available_at"] = pd.to_datetime(df["available_at"], utc=True, format='mixed')
    if (df["reference_time"] > df["available_at"]).any():
        raise ValueError("Observed-data references cannot postdate availability")
    if strict:
        if "pit_tier" not in df:
            raise ValueError("strict mode requires pit_tier")
        df = df[df["pit_tier"].eq("A")]
    df = df[df["available_at"] <= t.tz_convert("UTC")]
    version_key = keys + ["reference_time", "available_at"]
    if df.duplicated(version_key).any():
        raise ValueError("Ambiguous event versions; resolve duplicates before joining")
    df = df.sort_values(keys + ["reference_time", "available_at"])
    # A newly released revision to an old reference must not displace newer data.
    df = df.drop_duplicates(keys + ["reference_time"], keep="last")
    df = df.sort_values(keys + ["reference_time"]).drop_duplicates(keys, keep="last")
    df["reference_age_days"] = (t.tz_convert("UTC") - df["reference_time"]).dt.total_seconds()/86400
    df["release_age_days"] = (t.tz_convert("UTC") - df["available_at"]).dt.total_seconds()/86400
    return df.reset_index(drop=True)


def nport_external_flow(sales, redemptions) -> np.ndarray:
    """Reported sales minus redemptions, excluding separately reported reinvestment.

    Missing components remain NaN. Negative reported gross inputs are rejected.
    This is a measure on submitted data, not proof of organic external cash flow:
    mergers/liquidations and omnibus net reporting still require flags.
    """
    a, b = np.asarray(sales, dtype=float), np.asarray(redemptions, dtype=float)
    if a.shape != b.shape:
        raise ValueError("sales and redemptions must have matching shapes")
    if np.isinf(a).any() or np.isinf(b).any() or np.any(a < 0) or np.any(b < 0):
        raise ValueError("Gross flow components must be nonnegative finite or missing")
    return a - b


def mature_training_rows(frame: pd.DataFrame, train_cutoff) -> pd.DataFrame:
    """Only observations whose entire label interval and label publication matured."""
    required = {"decision_time", "label_end", "label_available_at"}
    if not required <= set(frame):
        raise ValueError("Missing label maturity columns")
    cutoff = pd.Timestamp(train_cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("train_cutoff must be timezone-aware")
    out = frame.copy()
    for col in required:
        out[col] = _aware_utc(out[col], col).to_numpy()
        out[col] = pd.to_datetime(out[col], utc=True)
    if (out["label_end"] <= out["decision_time"]).any():
        raise ValueError("Labels must end strictly after forecast origin")
    if (out["label_available_at"] < out["label_end"]).any():
        raise ValueError("Labels cannot be public before the target interval ends")
    ok = (out["label_end"] <= cutoff) & (out["label_available_at"] <= cutoff)
    return out.loc[ok].copy()
