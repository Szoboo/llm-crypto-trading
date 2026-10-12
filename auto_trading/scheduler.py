from __future__ import annotations

from datetime import datetime, timedelta, timezone


def next_4h_close_time(now: datetime | None = None, delay_seconds: int = 30) -> datetime:
    """Return the next UTC 4h candle close time plus a safety delay."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    base = now.replace(minute=0, second=0, microsecond=0)
    next_hour = ((base.hour // 4) + 1) * 4
    if next_hour >= 24:
        close_time = (base + timedelta(days=1)).replace(hour=0)
    else:
        close_time = base.replace(hour=next_hour)
    return close_time + timedelta(seconds=delay_seconds)


def seconds_until(target: datetime, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return max(0.0, (target - now).total_seconds())
