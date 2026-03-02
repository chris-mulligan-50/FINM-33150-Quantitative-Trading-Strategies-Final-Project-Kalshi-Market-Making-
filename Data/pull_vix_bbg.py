import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import blpapi


@dataclass
class BloombergConn:
    host: str = "localhost"
    port: int = 8194


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
    svc = session.getService("//blp/refdata")
    req = svc.createRequest("IntradayTickRequest")

    req.set("security", security)
    et = req.getElement("eventTypes")
    for e in event_types:
        et.appendValue(e)

    req.set("startDateTime", start_dt_utc)
    req.set("endDateTime", end_dt_utc)
    req.set("includeConditionCodes", include_condition_codes)
    req.set("includeNonPlottableEvents", False)

    session.sendRequest(req)

    rows = []
    done = False
    while not done:
        ev = session.nextEvent(5000)
        for msg in ev:
            if msg.hasElement("responseError"):
                err = msg.getElement("responseError")
                raise RuntimeError(f"Bloomberg responseError: {err}")

            if msg.messageType() == blpapi.Name("IntradayTickResponse"):
                tickData = msg.getElement("tickData").getElement("tickData")
                for i in range(tickData.numValues()):
                    t = tickData.getValueAsElement(i)
                    ts = t.getElementAsDatetime("time")
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
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _day_range(start_date: dt.date, end_date: dt.date):
    d = start_date
    one = dt.timedelta(days=1)
    while d <= end_date:
        yield d
        d += one


def vix_1s_series(
    start_date: dt.date,
    end_date: dt.date,
    out_parquet_name: Optional[str] = None,
    tz: str = "America/New_York",
    conn: BloombergConn = BloombergConn(),
) -> pd.DataFrame:
    session = _start_session(conn)

    chunks = []
    try:
        for day in _day_range(start_date, end_date):
            start_dt_utc = dt.datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=dt.timezone.utc)
            end_dt_utc = start_dt_utc + dt.timedelta(days=1)

            # VIX is an index; TRADE often empty. Try TRADE, then BID/ASK.
            df = _send_intraday_tick_request(
                session=session,
                security="VIX Index",
                start_dt_utc=start_dt_utc,
                end_dt_utc=end_dt_utc,
                event_types=["TRADE"],
            )

            if df.empty:
                df = _send_intraday_tick_request(
                    session=session,
                    security="VIX Index",
                    start_dt_utc=start_dt_utc,
                    end_dt_utc=end_dt_utc,
                    event_types=["BID", "ASK"],
                )

            if df.empty:
                continue

            df = df.sort_values("time").set_index("time")

            by_type = {}
            for e_type, sub in df.groupby("event_type"):
                by_type[e_type] = sub["value"].resample("1s").last()

            # Prefer TRADE if present; else mid(BID,ASK); else last available stream
            if "TRADE" in by_type:
                px_1s = by_type["TRADE"]
            else:
                bid = by_type.get("BID")
                ask = by_type.get("ASK")
                if bid is not None and ask is not None:
                    px_1s = (bid + ask) / 2.0
                else:
                    px_1s = next(iter(by_type.values()))

            px_1s = px_1s.ffill().tz_convert(tz)
            chunks.append(px_1s.to_frame("price"))

    finally:
        session.stop()

    if not chunks:
        return pd.DataFrame(columns=["price"], index=pd.DatetimeIndex([], tz=tz, name="time"))

    out = pd.concat(chunks).sort_index()
    out.index.name = "time"

    if out_parquet_name:
        data_dir = Path("_data")
        data_dir.mkdir(exist_ok=True)
        out_path = data_dir / out_parquet_name
        out.to_parquet(out_path)
        print(f"Saved: {out_path.resolve()}")

    return out


if __name__ == "__main__":
    start = dt.date(2026, 1, 1)
    end = dt.date(2026, 2, 26)

    df_vix = vix_1s_series(
        start_date=start,
        end_date=end,
        out_parquet_name="vix_1s_2026-01-01_to_2026-02-26.parquet",
        tz="America/New_York",
    )

    print(df_vix.head())
    print(df_vix.tail())
    print("Rows:", len(df_vix))