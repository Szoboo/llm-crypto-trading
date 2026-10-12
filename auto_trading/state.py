from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone


@dataclass
class BotState:
    last_candle_timestamp: str | None = None
    last_order_id: str | None = None
    last_decision: str = "HOLD"

    position_open: bool = False
    entry_price_krw: float | None = None
    entry_amount_eth: float | None = None
    entry_notional_krw: float | None = None
    stop_loss_krw: float | None = None
    take_profit_krw: float | None = None
    position_opened_at: str | None = None
    last_exit_order_id: str | None = None
    last_exit_reason: str | None = None

    updated_at: str | None = None


def load_state(path: str) -> BotState:
    if not os.path.exists(path):
        return BotState()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    allowed = {field.name for field in fields(BotState)}
    clean = {k: v for k, v in data.items() if k in allowed}
    return BotState(**clean)


def save_state(path: str, state: BotState) -> None:
    state.updated_at = datetime.now(timezone.utc).isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(state), f, ensure_ascii=False, indent=2)
