from datetime import datetime, timezone



from auto_trading.scheduler import next_4h_close_time





def test_next_4h_close_time():

    now = datetime(2026, 5, 14, 1, 23, tzinfo=timezone.utc)

    target = next_4h_close_time(now=now, delay_seconds=30)

    assert target.hour == 4

    assert target.minute == 0

    assert target.second == 30

