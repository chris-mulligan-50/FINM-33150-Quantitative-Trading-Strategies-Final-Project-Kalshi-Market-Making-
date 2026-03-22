"""
Pull 1-second series for SPX Index from Bloomberg Desktop API (blpapi)
by requesting intraday ticks and resampling to 1-second.

Requirements:
  - Bloomberg Terminal running + API entitlement
  - blpapi installed (pip install blpapi)
  - pandas installed

Notes:
  - Bloomberg IntradayTickRequest returns ticks (irregular times); we resample to 1-second.
  - Indexes may not have TRADE ticks; script falls back to BID/ASK if TRADE empty.
"""

import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import blpapi
from pathlib import Path

DATA_DIR = Path("_data")
DATA_DIR.mkdir(exist_ok=True)

@dataclass
class BloombergConn:
    host: str = "localhost"
    port: int = 8194  # standard Desktop API port


def _start_session(conn: BloombergConn) -> blpapi.Session:
    opts = blpapi.SessionOptions()
    opts.setServerHost(conn.host)
    opts.setServerPort(conn.port)

    session = blpapi.Session(opts)
    if not session.start():
        raise RuntimeError("Failed to start blpapi session. Is Bloomberg Terminal running?")

    if not session.openService("//blp/refdata"):
        raise RuntimeError("Failed to open //blp/refdata service.")

    return session


def _send_intraday_tick_request(
    session: blpapi.Session,
    security: str,
    start_dt_utc: dt.datetime,
    end_dt_utc: dt.datetime,
    event_types: List[str],
    include_condition_codes: bool = False,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      time (datetime, UTC), event_type (str), value (float), size (float|None)
    """
    svc = session.getService("//blp/refdata")
    req = svc.createRequest("IntradayTickRequest")

    req.set("security", security)

    # eventTypes is an array
    et = req.getElement("eventTypes")
    for e in event_types:
        et.appendValue(e)

    req.set("startDateTime", start_dt_utc)
    req.set("endDateTime", end_dt_utc)

    # Common options
    req.set("includeConditionCodes", include_condition_codes)
    req.set("includeNonPlottableEvents", False)

    session.sendRequest(req)

    rows = []
    done = False

    while not done:
        ev = session.nextEvent(5000)  # ms
        for msg in ev:
            # ResponseError check
            if msg.hasElement("responseError"):
                err = msg.getElement("responseError")
                raise RuntimeError(f"Bloomberg responseError: {err}")

            if msg.messageType() == blpapi.Name("IntradayTickResponse"):
                tickData = msg.getElement("tickData").getElement("tickData")
                for i in range(tickData.numValues()):
                    t = tickData.getValueAsElement(i)
                    ts = t.getElementAsDatetime("time")  # timezone-aware in blpapi
                    e_type = t.getElementAsString("type")
                    val = t.getElementAsFloat("value")
                    size = None
                    if t.hasElement("size"):
                        try:
                            size = t.getElementAsFloat("size")
                        except Exception:
                            size = None
                    rows.append((ts, e_type, val, size))

        if ev.eventType() == blpapi.Event.RESPONSE:
            done = True

    df = pd.DataFrame(rows, columns=["time", "event_type", "value", "size"])
    if not df.empty:
        # Normalize to pandas datetime with UTC tz
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _day_range(start_date: dt.date, end_date: dt.date):
    d = start_date
    one = dt.timedelta(days=1)
    while d <= end_date:
        yield d
        d += one


def spx_1s_series(
    start_date: dt.date,
    end_date: dt.date,
    conn: BloombergConn = BloombergConn(),
    out_parquet: Optional[str] = DATA_DIR / "spx_1s.parquet",
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """
    Pull SPX Index ticks day-by-day and resample to 1-second.
    Returns DataFrame indexed by second (tz-aware, converted to tz argument),
    with a single column: price

    Writes parquet if out_parquet is not None.
    """
    session = _start_session(conn)

    all_seconds = []

    try:
        for day in _day_range(start_date, end_date):
            # Pull full calendar day in UTC. You can tighten to US session later if desired.
            start_dt_utc = dt.datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=dt.timezone.utc)
            end_dt_utc = start_dt_utc + dt.timedelta(days=1)

            # 1) Try TRADE ticks first (may be empty for indices)
            df = _send_intraday_tick_request(
                session=session,
                security="SPX Index",
                start_dt_utc=start_dt_utc,
                end_dt_utc=end_dt_utc,
                event_types=["TRADE"],
            )

            # 2) Fallback: BID/ASK if TRADE empty
            if df.empty:
                df = _send_intraday_tick_request(
                    session=session,
                    security="SPX Index",
                    start_dt_utc=start_dt_utc,
                    end_dt_utc=end_dt_utc,
                    event_types=["BID", "ASK"],
                )

            if df.empty:
                # holiday/weekend/no entitlement/no ticks
                continue

            # Choose a "price" from ticks:
            # - if only BID/ASK exist, take mid when both present in same second, else last tick.
            df = df.sort_values("time")
            df = df.set_index("time")

            # Resample each event_type to 1s last
            by_type = {}
            for e_type, sub in df.groupby("event_type"):
                by_type[e_type] = sub["value"].resample("1s").last()

            if "TRADE" in by_type:
                px_1s = by_type["TRADE"]
            else:
                bid = by_type.get("BID")
                ask = by_type.get("ASK")
                if bid is not None and ask is not None:
                    px_1s = (bid + ask) / 2.0
                else:
                    # last available stream
                    px_1s = next(iter(by_type.values()))

            # Optional: forward-fill within the day so every second has a value after first tick
            px_1s = px_1s.ffill()

            # Convert timezone (still indexed in UTC before conversion)
            px_1s = px_1s.tz_convert(tz)

            day_df = px_1s.to_frame(name="price")
            all_seconds.append(day_df)

    finally:
        session.stop()

    if not all_seconds:
        return pd.DataFrame(columns=["price"]).set_index(pd.DatetimeIndex([], tz=tz, name="time"))

    out = pd.concat(all_seconds).sort_index()
    out.index.name = "time"

    if out_parquet:
        out.to_parquet(out_parquet)

    return out


if __name__ == "__main__":
    start = dt.date(2026, 1, 1)
    end = dt.date(2026, 2, 26)

    df_spx = spx_1s_series(
        start_date=start,
        end_date=end,
        out_parquet="spx_1s_2026-01-01_to_2026-02-26.parquet",
        tz="America/New_York",
    )

    print(df_spx.head())
    print(df_spx.tail())
    print("Rows:", len(df_spx))