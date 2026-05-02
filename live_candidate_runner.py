from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import websockets

from btcusdt_usdm_futures_bot import (
    BinanceFuturesClient,
    RuntimeConfig,
    StrategyConfig,
    build_market_order_params,
    evaluate_latest_signal,
    extract_symbol_filters,
    parse_kline_stream,
    quantize_down,
)


SYMBOL = "BTCUSDT"
INTERVAL = "1h"
REQUIRED_SIGNAL = -1
USER_STREAM_KEEPALIVE_SECONDS = 30 * 60
USER_STREAM_STALE_SECONDS = 90

logger = logging.getLogger("live_candidate")


@dataclass(frozen=True)
class LiveCandidateConfig:
    mode: str
    testnet: bool
    dry_run: bool
    allocated_capital_usdt: Decimal
    max_notional_usdt: Decimal
    max_capital_pct_per_trade: Decimal
    max_positions: int
    max_daily_loss_pct: Decimal
    max_cumulative_loss_pct: Decimal
    test_order_notional_usdt: Decimal
    leverage: int
    margin_type: str
    log_file: Path
    require_flat_position: bool = True
    allow_live_env_var: str = "LIVE_TRADING"
    allow_demo_env_var: str = "DEMO_TRADING"


@dataclass
class ExchangeState:
    listen_key: Optional[str] = None
    user_stream_connected: bool = False
    last_user_event_at: Optional[float] = None
    position_amt: Decimal = Decimal("0")
    entry_price: Decimal = Decimal("0")
    last_order_update: Optional[dict[str, Any]] = None
    last_account_update: Optional[dict[str, Any]] = None
    open_order_count: int = 0
    wallet_balance_usdt: Optional[Decimal] = None
    available_balance_usdt: Optional[Decimal] = None
    private_api_available: bool = False
    private_api_error: Optional[str] = None
    daily_realized_pnl: Decimal = Decimal("0")
    cumulative_realized_pnl: Decimal = Decimal("0")

    @property
    def is_flat(self) -> bool:
        return self.position_amt == 0


class JsonEventLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type: str, **payload: Any) -> None:
        event = {
            "ts": pd.Timestamp.now(tz="UTC").isoformat(),
            "event_type": event_type,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str, sort_keys=True) + "\n")
        logger.info("%s | %s", event_type, payload)


class GuardrailViolation(RuntimeError):
    pass


class LiveCandidateRunner:
    def __init__(self, config: LiveCandidateConfig):
        runtime = RuntimeConfig(symbol=SYMBOL, interval=INTERVAL, testnet=config.testnet, dry_run=config.dry_run)
        self.runtime = runtime
        self.config = config
        self.client = BinanceFuturesClient(runtime)
        self.state = ExchangeState()
        self.event_logger = JsonEventLogger(config.log_file)
        self.strategy = StrategyConfig(allow_long=False, allow_short=True)
        self.candles: list[dict[str, Any]] = []
        self.latest_mark_price: Optional[Decimal] = None
        self.last_order_signal_ts: Optional[str] = None
        self.session_day = pd.Timestamp.now(tz="UTC").date()

    async def run(self) -> None:
        self.validate_static_guardrails()
        self.private_api_preflight_or_abort()
        await self.reconcile_or_abort()
        self.prepare_account_settings_or_abort()

        if self.config.mode == "test-order":
            self.submit_test_order_or_abort()
            return

        await self.start_user_data_stream()
        tasks = [
            asyncio.create_task(self.market_data_loop()),
            asyncio.create_task(self.user_data_loop()),
            asyncio.create_task(self.listen_key_keepalive_loop()),
            asyncio.create_task(self.stream_health_loop()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await self.close_listen_key()

    def private_api_preflight_or_abort(self) -> None:
        if not os.getenv("BINANCE_FUTURES_API_KEY"):
            if self.config.mode == "observe":
                self.event_logger.emit("private_api_preflight_skipped", reason="observe_without_api_key")
                return
            raise GuardrailViolation("Private API preflight requires BINANCE_FUTURES_API_KEY")

        try:
            self.client.account()
            self.state.private_api_available = True
            self.event_logger.emit("private_api_preflight_passed")
        except Exception as exc:
            self.state.private_api_available = False
            self.state.private_api_error = str(exc)
            error_text = str(exc)
            if "private key file not found" in error_text.lower() or "Missing BINANCE_FUTURES_PRIVATE_KEY_PATH" in error_text:
                likely_block = "missing_or_wrong_local_ed25519_private_key_path"
            elif "Invalid API-key, IP, or permissions" in error_text:
                likely_block = "invalid_key_or_permissions_or_ip_whitelist_or_wrong_environment"
            else:
                likely_block = "private_api_unavailable"
            self.event_logger.emit(
                "private_api_preflight_failed",
                error=error_text,
                likely_block=likely_block,
            )
            if self.config.mode != "observe":
                raise GuardrailViolation(f"Private API preflight failed: {exc}") from exc

    def validate_static_guardrails(self) -> None:
        if self.runtime.symbol != SYMBOL or self.runtime.interval != INTERVAL:
            raise GuardrailViolation("Runner is hard-restricted to BTCUSDT 1h")
        if self.config.allocated_capital_usdt <= 0:
            raise GuardrailViolation("ALLOCATED_CAPITAL_USDT must be positive")
        if self.config.max_notional_usdt <= 0:
            raise GuardrailViolation("MAX_NOTIONAL_USDT must be positive")
        if self.config.max_capital_pct_per_trade <= 0 or self.config.max_capital_pct_per_trade > 100:
            raise GuardrailViolation("MAX_CAPITAL_PCT_PER_TRADE must be > 0 and <= 100")
        if self.config.max_positions < 1:
            raise GuardrailViolation("MAX_POSITIONS must be at least 1")
        if self.config.max_positions != 1:
            raise GuardrailViolation("This runner is currently restricted to MAX_POSITIONS=1")
        if self.config.max_daily_loss_pct <= 0 or self.config.max_cumulative_loss_pct <= 0:
            raise GuardrailViolation("Loss limits must be positive percentages")

        if self.config.mode == "demo-supervised":
            if not self.config.testnet:
                raise GuardrailViolation("demo-supervised is restricted to Binance Futures demo/testnet")
            if self.config.dry_run:
                self.event_logger.emit("demo_supervised_dry_run", reason="demo orders require --allow-demo-orders and DEMO_TRADING=true")
            elif os.getenv(self.config.allow_demo_env_var, "").lower() != "true":
                raise GuardrailViolation(f"Set {self.config.allow_demo_env_var}=true to permit demo/testnet orders")

        if self.config.mode == "supervised-live":
            if self.config.dry_run:
                raise GuardrailViolation("supervised-live cannot run with dry_run=True")
            if self.config.testnet:
                raise GuardrailViolation("supervised-live is for live endpoint only; use observe/test-order first")
            if os.getenv(self.config.allow_live_env_var, "").lower() != "true":
                raise GuardrailViolation(f"Set {self.config.allow_live_env_var}=true to permit live endpoint")
            if os.getenv("LIVE_TRADING_MANUAL_CONFIRMATION", "") != "BTCUSDT_1H_SHORT_ONLY":
                raise GuardrailViolation("Missing LIVE_TRADING_MANUAL_CONFIRMATION=BTCUSDT_1H_SHORT_ONLY")

        if self.config.mode in {"test-order", "demo-supervised", "supervised-live"}:
            auth_type = os.getenv("BINANCE_FUTURES_AUTH_TYPE", "HMAC").strip().upper()
            if not os.getenv("BINANCE_FUTURES_API_KEY"):
                raise GuardrailViolation("Missing BINANCE_FUTURES_API_KEY")
            if auth_type == "HMAC" and not os.getenv("BINANCE_FUTURES_API_SECRET"):
                raise GuardrailViolation("Missing BINANCE_FUTURES_API_SECRET for HMAC authentication")
            if auth_type == "ED25519" and not os.getenv("BINANCE_FUTURES_PRIVATE_KEY_PATH"):
                raise GuardrailViolation("Missing BINANCE_FUTURES_PRIVATE_KEY_PATH for Ed25519 authentication")

        self.event_logger.emit(
            "guardrails_static_passed",
            mode=self.config.mode,
            dry_run=self.config.dry_run,
            testnet=self.config.testnet,
            auth_type=os.getenv("BINANCE_FUTURES_AUTH_TYPE", "HMAC").strip().upper(),
            allocated_capital=str(self.config.allocated_capital_usdt),
            max_notional=str(self.config.max_notional_usdt),
            max_capital_pct_per_trade=str(self.config.max_capital_pct_per_trade),
            max_positions=self.config.max_positions,
            max_daily_loss_pct=str(self.config.max_daily_loss_pct),
            max_cumulative_loss_pct=str(self.config.max_cumulative_loss_pct),
            leverage=self.config.leverage,
            margin_type=self.config.margin_type,
        )

    async def reconcile_or_abort(self) -> None:
        if self.config.mode == "observe" and not self.state.private_api_available:
            self.event_logger.emit("reconciliation_skipped", reason=self.state.private_api_error or "private_api_unavailable")
            return
        if self.config.mode == "observe" and not os.getenv("BINANCE_FUTURES_API_KEY"):
            self.event_logger.emit("reconciliation_skipped", reason="observe_without_api_key")
            return

        try:
            positions = self.client.position_risk(SYMBOL)
        except Exception as exc:
            if self.config.mode == "observe":
                self.event_logger.emit("reconciliation_skipped", reason=str(exc))
                return
            raise GuardrailViolation(f"Could not reconcile position risk: {exc}") from exc

        if not positions:
            self.state.position_amt = Decimal("0")
            self.state.entry_price = Decimal("0")
            self.event_logger.emit("position_reconciled", position_amt="0", entry_price="0", source="empty_position_risk")
            self.validate_virtual_capital_limits_or_abort(mark_price=None)
            positions = []

        try:
            account = self.client.account()
            self.state.wallet_balance_usdt = Decimal(str(account.get("totalWalletBalance", "0")))
            self.state.available_balance_usdt = Decimal(str(account.get("availableBalance", "0")))
            self.event_logger.emit(
                "account_balance_observed_not_used_for_sizing",
                wallet_balance=str(self.state.wallet_balance_usdt),
                available_balance=str(self.state.available_balance_usdt),
                allocated_capital=str(self.config.allocated_capital_usdt),
            )
        except Exception as exc:
            if self.config.mode == "observe":
                self.event_logger.emit("account_balance_observation_skipped", reason=str(exc))
            else:
                raise GuardrailViolation(f"Could not observe account balance: {exc}") from exc

        if positions:
            position = positions[0]
            self.state.position_amt = Decimal(str(position.get("positionAmt", "0")))
            self.state.entry_price = Decimal(str(position.get("entryPrice", "0")))
            self.event_logger.emit(
                "position_reconciled",
                position_amt=str(self.state.position_amt),
                entry_price=str(self.state.entry_price),
                source="position_risk",
            )

        if self.config.require_flat_position and not self.state.is_flat:
            raise GuardrailViolation(f"Expected flat position before start, got {self.state.position_amt}")

        self.validate_virtual_capital_limits_or_abort(mark_price=None)

        try:
            open_orders = self.client.open_orders(SYMBOL)
        except Exception as exc:
            if self.config.mode == "observe":
                self.event_logger.emit("open_orders_reconciliation_skipped", reason=str(exc))
                return
            raise GuardrailViolation(f"Could not reconcile open orders: {exc}") from exc

        self.state.open_order_count = len(open_orders)
        self.event_logger.emit("open_orders_reconciled", open_order_count=self.state.open_order_count)
        if self.state.open_order_count:
            raise GuardrailViolation(f"Expected no open orders before start, got {self.state.open_order_count}")

    def prepare_account_settings_or_abort(self) -> None:
        if self.config.mode not in {"test-order", "demo-supervised", "supervised-live"}:
            return
        if self.config.leverage < 1 or self.config.leverage > 125:
            raise GuardrailViolation(f"Invalid leverage: {self.config.leverage}")

        try:
            leverage_response = self.client.change_leverage(SYMBOL, self.config.leverage)
            self.event_logger.emit("leverage_checked", response=leverage_response)
        except Exception as exc:
            raise GuardrailViolation(f"Could not set/check leverage: {exc}") from exc

        if self.config.margin_type:
            margin_type = self.config.margin_type.upper()
            if margin_type not in {"ISOLATED", "CROSSED"}:
                raise GuardrailViolation(f"Invalid margin_type: {self.config.margin_type}")
            try:
                response = self.client.change_margin_type(SYMBOL, margin_type)
                self.event_logger.emit("margin_type_checked", response=response)
            except Exception as exc:
                message = str(exc)
                if "No need to change margin type" in message or "-4046" in message:
                    self.event_logger.emit("margin_type_already_set", margin_type=margin_type)
                else:
                    raise GuardrailViolation(f"Could not set/check margin type: {exc}") from exc

    def submit_test_order_or_abort(self) -> None:
        filters = extract_symbol_filters(self.client.exchange_info(), SYMBOL)
        mark = Decimal(self.client.mark_price(SYMBOL)["markPrice"])
        order_notional = min(self.config.test_order_notional_usdt, self.compute_allowed_order_notional())
        self.validate_virtual_capital_limits_or_abort(mark_price=mark, proposed_order_notional=order_notional)
        qty = quantize_down(order_notional / mark, filters.step_size)
        if qty < filters.min_qty or qty * mark < filters.min_notional:
            raise GuardrailViolation(f"Test order too small after filters: qty={qty}, notional={qty * mark}")

        params = build_market_order_params(SYMBOL, REQUIRED_SIGNAL, qty)
        self.event_logger.emit(
            "test_order_submit",
            params=sanitize_order_params(params),
            mark_price=str(mark),
            allocated_capital=str(self.config.allocated_capital_usdt),
            order_notional=str(qty * mark),
            max_order_notional=str(order_notional),
        )
        response = self.client.test_order(params)
        self.event_logger.emit("test_order_accepted", response=response)

    async def start_user_data_stream(self) -> None:
        if self.config.mode == "observe" and not self.state.private_api_available:
            self.event_logger.emit("user_stream_skipped", reason=self.state.private_api_error or "private_api_unavailable")
            return
        if not os.getenv("BINANCE_FUTURES_API_KEY"):
            if self.config.mode == "observe":
                self.event_logger.emit("user_stream_skipped", reason="observe_without_api_key")
                return
            raise GuardrailViolation("User data stream requires API key")
        listen_key = self.client.start_listen_key()
        self.state.listen_key = listen_key
        self.state.last_user_event_at = time.time()
        self.event_logger.emit("listen_key_started", listen_key_masked=mask_secret(listen_key))

    async def close_listen_key(self) -> None:
        if self.state.listen_key:
            try:
                self.client.close_listen_key(self.state.listen_key)
                self.event_logger.emit("listen_key_closed")
            except Exception as exc:
                self.event_logger.emit("listen_key_close_error", error=str(exc))

    async def listen_key_keepalive_loop(self) -> None:
        while True:
            await asyncio.sleep(USER_STREAM_KEEPALIVE_SECONDS)
            if not self.state.listen_key:
                continue
            try:
                self.client.keepalive_listen_key(self.state.listen_key)
                self.event_logger.emit("listen_key_keepalive")
            except Exception as exc:
                self.event_logger.emit("listen_key_keepalive_error", error=str(exc))
                raise

    async def user_data_loop(self) -> None:
        if not self.state.listen_key:
            return
        url = f"{self.runtime.user_ws_base_url}/{self.state.listen_key}"
        while True:
            try:
                async with websockets.connect(url, ping_interval=180, ping_timeout=600) as websocket:
                    self.state.user_stream_connected = True
                    self.event_logger.emit("user_stream_connected")
                    async for raw in websocket:
                        self.state.last_user_event_at = time.time()
                        event = json.loads(raw)
                        self.handle_user_event(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.state.user_stream_connected = False
                self.event_logger.emit("user_stream_error", error=str(exc))
                if self.config.mode != "observe":
                    raise GuardrailViolation("User data stream failed") from exc
                await asyncio.sleep(5)

    def handle_user_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("e")
        if event_type == "ACCOUNT_UPDATE":
            self.state.last_account_update = event
            for position in event.get("a", {}).get("P", []):
                if position.get("s") == SYMBOL:
                    self.state.position_amt = Decimal(position.get("pa", "0"))
                    self.state.entry_price = Decimal(position.get("ep", "0"))
            self.reset_daily_pnl_if_needed()
            self.event_logger.emit(
                "account_update",
                position_amt=str(self.state.position_amt),
                entry_price=str(self.state.entry_price),
                daily_realized_pnl=str(self.state.daily_realized_pnl),
                cumulative_realized_pnl=str(self.state.cumulative_realized_pnl),
            )
        elif event_type == "ORDER_TRADE_UPDATE":
            self.state.last_order_update = event
            order = event.get("o", {})
            realized_pnl = Decimal(str(order.get("rp", "0")))
            if realized_pnl:
                self.reset_daily_pnl_if_needed()
                self.state.daily_realized_pnl += realized_pnl
                self.state.cumulative_realized_pnl += realized_pnl
            self.event_logger.emit(
                "order_trade_update",
                client_order_id=order.get("c"),
                side=order.get("S"),
                status=order.get("X"),
                execution_type=order.get("x"),
                filled_qty=order.get("z"),
                avg_price=order.get("ap"),
                realized_pnl=str(realized_pnl),
                daily_realized_pnl=str(self.state.daily_realized_pnl),
                cumulative_realized_pnl=str(self.state.cumulative_realized_pnl),
            )
        else:
            self.event_logger.emit("user_event", event=event)

    async def stream_health_loop(self) -> None:
        while True:
            await asyncio.sleep(15)
            if self.config.mode == "observe" and not self.state.listen_key:
                continue
            if not self.state.last_user_event_at:
                continue
            age = time.time() - self.state.last_user_event_at
            if age > USER_STREAM_STALE_SECONDS:
                self.event_logger.emit("user_stream_stale", age_seconds=age)
                if self.config.mode != "observe":
                    raise GuardrailViolation("User data stream stale")

    async def market_data_loop(self) -> None:
        while True:
            try:
                async with websockets.connect(self.runtime.market_ws_url, ping_interval=180, ping_timeout=600) as websocket:
                    self.event_logger.emit("market_stream_connected", url=self.runtime.market_ws_url)
                    async for raw in websocket:
                        payload = json.loads(raw)
                        data = payload.get("data", payload)
                        event_type = data.get("e")
                        if event_type == "markPriceUpdate":
                            self.latest_mark_price = Decimal(str(data["p"]))
                            continue
                        candle = parse_kline_stream(data)
                        if candle is None or not candle["is_closed"]:
                            continue
                        self.candles.append(candle)
                        self.candles = self.candles[-600:]
                        signal = evaluate_latest_signal(self.candles, SYMBOL, self.strategy)
                        self.event_logger.emit(
                            "signal_evaluated",
                            candle_time=str(candle.get("timestamp")),
                            close=str(candle["close"]),
                            mark_price=str(self.latest_mark_price) if self.latest_mark_price else None,
                            signal=signal,
                            candles=len(self.candles),
                        )
                        await self.handle_signal(signal, str(candle.get("timestamp")))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.event_logger.emit("market_stream_error", error=str(exc))
                if self.config.mode != "observe":
                    raise
                await asyncio.sleep(5)

    async def handle_signal(self, signal: int, signal_ts: str) -> None:
        if signal != REQUIRED_SIGNAL:
            return
        self.validate_virtual_capital_limits_or_abort(mark_price=self.latest_mark_price)
        if self.last_order_signal_ts == signal_ts:
            self.event_logger.emit("signal_blocked", reason="duplicate_signal_candle", candle_time=signal_ts)
            return
        if not self.state.is_flat:
            self.event_logger.emit("signal_blocked", reason="not_flat", position_amt=str(self.state.position_amt))
            return
        self.event_logger.emit("signal_decision", action="short_candidate", candle_time=signal_ts, dry_run=self.config.dry_run)
        if self.config.mode == "observe":
            self.event_logger.emit("paper_signal", action="would_submit_short", dry_run=True)
            return
        if self.config.mode == "test-order":
            self.submit_test_order_or_abort()
            self.last_order_signal_ts = signal_ts
            return
        if self.config.mode == "demo-supervised":
            await self.submit_demo_supervised_order(signal_ts)
            return
        if self.config.mode == "supervised-live":
            await self.submit_supervised_live_order(signal_ts)

    async def submit_demo_supervised_order(self, signal_ts: str) -> None:
        if self.config.dry_run:
            self.event_logger.emit("demo_order_blocked", reason="dry_run", candle_time=signal_ts)
            return
        await self.submit_exchange_order(signal_ts=signal_ts, event_prefix="demo_order")

    async def submit_supervised_live_order(self, signal_ts: str) -> None:
        if self.config.dry_run:
            raise GuardrailViolation("dry_run still enabled")
        await self.submit_exchange_order(signal_ts=signal_ts, event_prefix="live_order")

    async def submit_exchange_order(self, signal_ts: str, event_prefix: str) -> None:
        if not self.latest_mark_price:
            raise GuardrailViolation("No mark price available")
        if not self.state.user_stream_connected:
            raise GuardrailViolation("User data stream not connected")

        filters = extract_symbol_filters(self.client.exchange_info(), SYMBOL)
        order_notional = self.compute_allowed_order_notional()
        self.validate_virtual_capital_limits_or_abort(mark_price=self.latest_mark_price, proposed_order_notional=order_notional)
        qty = quantize_down(order_notional / self.latest_mark_price, filters.step_size)
        if qty < filters.min_qty or qty * self.latest_mark_price < filters.min_notional:
            raise GuardrailViolation(f"Live quantity too small: qty={qty}")

        params = build_market_order_params(SYMBOL, REQUIRED_SIGNAL, qty)
        self.event_logger.emit(
            f"{event_prefix}_submit_attempt",
            params=sanitize_order_params(params),
            candle_time=signal_ts,
            allocated_capital=str(self.config.allocated_capital_usdt),
            order_notional=str(qty * self.latest_mark_price),
            max_order_notional=str(order_notional),
        )
        response = self.client.new_order(params)
        self.last_order_signal_ts = signal_ts
        self.event_logger.emit(f"{event_prefix}_response", response=response, candle_time=signal_ts)

    def reset_daily_pnl_if_needed(self) -> None:
        today = pd.Timestamp.now(tz="UTC").date()
        if today != self.session_day:
            self.session_day = today
            self.state.daily_realized_pnl = Decimal("0")
            self.event_logger.emit("daily_risk_window_reset", session_day=str(today))

    def current_position_notional(self, mark_price: Optional[Decimal]) -> Decimal:
        if self.state.position_amt == 0:
            return Decimal("0")
        price = mark_price or self.state.entry_price
        return abs(self.state.position_amt) * price

    def allocated_equity(self) -> Decimal:
        return self.config.allocated_capital_usdt + self.state.cumulative_realized_pnl

    def daily_loss_limit_usdt(self) -> Decimal:
        return self.config.allocated_capital_usdt * self.config.max_daily_loss_pct / Decimal("100")

    def cumulative_loss_limit_usdt(self) -> Decimal:
        return self.config.allocated_capital_usdt * self.config.max_cumulative_loss_pct / Decimal("100")

    def compute_allowed_order_notional(self) -> Decimal:
        pct_notional = self.config.allocated_capital_usdt * self.config.max_capital_pct_per_trade / Decimal("100")
        return min(self.config.max_notional_usdt, pct_notional, self.allocated_equity())

    def validate_virtual_capital_limits_or_abort(
        self,
        mark_price: Optional[Decimal],
        proposed_order_notional: Decimal = Decimal("0"),
    ) -> None:
        self.reset_daily_pnl_if_needed()
        allocated_equity = self.allocated_equity()
        current_notional = self.current_position_notional(mark_price)
        projected_notional = current_notional + proposed_order_notional
        required_margin = projected_notional / Decimal(str(self.config.leverage))

        if allocated_equity <= 0:
            raise GuardrailViolation("Allocated equity depleted; blocking new risk")
        if self.state.daily_realized_pnl <= -self.daily_loss_limit_usdt():
            raise GuardrailViolation("Daily loss limit reached on allocated capital")
        if self.state.cumulative_realized_pnl <= -self.cumulative_loss_limit_usdt():
            raise GuardrailViolation("Cumulative loss limit reached on allocated capital")
        if projected_notional > self.config.allocated_capital_usdt:
            raise GuardrailViolation(
                f"Projected notional {projected_notional} exceeds allocated capital {self.config.allocated_capital_usdt}"
            )
        if required_margin > allocated_equity:
            raise GuardrailViolation(f"Required margin {required_margin} exceeds allocated equity {allocated_equity}")

        self.event_logger.emit(
            "virtual_capital_guardrails_passed",
            allocated_capital=str(self.config.allocated_capital_usdt),
            allocated_equity=str(allocated_equity),
            current_notional=str(current_notional),
            proposed_order_notional=str(proposed_order_notional),
            projected_notional=str(projected_notional),
            required_margin=str(required_margin),
            daily_realized_pnl=str(self.state.daily_realized_pnl),
            cumulative_realized_pnl=str(self.state.cumulative_realized_pnl),
        )


def mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def sanitize_order_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if key.lower() not in {"signature"}}


def parse_bool(value: str, default: bool = False) -> bool:
    if value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if value in {"", "replace_me"}:
            continue
        if key and key not in os.environ:
            os.environ[key] = value


def apply_demo_key_aliases(testnet: bool) -> None:
    if not testnet:
        return
    key_aliases = ("BINANCE_FUTURES_DEMO_API_KEY", "BINANCE_FUTURES_TESTNET_API_KEY")
    secret_aliases = ("BINANCE_FUTURES_DEMO_API_SECRET", "BINANCE_FUTURES_TESTNET_API_SECRET")
    private_key_aliases = ("BINANCE_FUTURES_DEMO_PRIVATE_KEY_PATH", "BINANCE_FUTURES_TESTNET_PRIVATE_KEY_PATH")
    private_key_passphrase_aliases = (
        "BINANCE_FUTURES_DEMO_PRIVATE_KEY_PASSPHRASE",
        "BINANCE_FUTURES_TESTNET_PRIVATE_KEY_PASSPHRASE",
    )
    if not os.getenv("BINANCE_FUTURES_API_KEY"):
        for name in key_aliases:
            if os.getenv(name):
                os.environ["BINANCE_FUTURES_API_KEY"] = os.environ[name]
                break
    if not os.getenv("BINANCE_FUTURES_API_SECRET"):
        for name in secret_aliases:
            if os.getenv(name):
                os.environ["BINANCE_FUTURES_API_SECRET"] = os.environ[name]
                break
    if not os.getenv("BINANCE_FUTURES_PRIVATE_KEY_PATH"):
        for name in private_key_aliases:
            if os.getenv(name):
                os.environ["BINANCE_FUTURES_PRIVATE_KEY_PATH"] = os.environ[name]
                break
    if not os.getenv("BINANCE_FUTURES_PRIVATE_KEY_PASSPHRASE"):
        for name in private_key_passphrase_aliases:
            if os.getenv(name):
                os.environ["BINANCE_FUTURES_PRIVATE_KEY_PASSPHRASE"] = os.environ[name]
                break


def decimal_from_env(name: str, fallback: str) -> Decimal:
    return Decimal(str(os.getenv(name, fallback)))


def int_from_env(name: str, fallback: str) -> int:
    return int(os.getenv(name, fallback))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Restricted BTCUSDT USD-M 1h short-only live-candidate runner")
    parser.add_argument("--mode", choices=["observe", "test-order", "demo-supervised", "supervised-live"], default="observe")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Optional .env file. Existing environment variables win.")
    parser.add_argument("--testnet", action="store_true", default=True)
    parser.add_argument("--mainnet", action="store_true", help="Use mainnet endpoint. Live still requires hard guardrails.")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--allow-demo-orders", action="store_true", help="Only meaningful with demo-supervised plus DEMO_TRADING=true.")
    parser.add_argument("--allow-non-dry-run", action="store_true", help="Only meaningful with supervised-live and live env vars.")
    parser.add_argument("--symbol", default=os.getenv("FUTURES_SYMBOL", SYMBOL))
    parser.add_argument("--interval", default=os.getenv("FUTURES_INTERVAL", INTERVAL))
    parser.add_argument("--allocated-capital-usdt", default=None)
    parser.add_argument("--max-notional-usdt", default=None)
    parser.add_argument("--max-capital-pct-per-trade", default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--max-daily-loss-pct", default=None)
    parser.add_argument("--max-cumulative-loss-pct", default=None)
    parser.add_argument("--test-order-notional-usdt", default=None)
    parser.add_argument("--leverage", type=int, default=None)
    parser.add_argument("--margin-type", default=None, choices=["ISOLATED", "CROSSED", "isolated", "crossed"])
    parser.add_argument("--log-file", type=Path, default=Path("logs/live_candidate_events.jsonl"))
    parser.add_argument("--i-understand-live-risk", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    args = build_arg_parser().parse_args()
    load_env_file(args.env_file)
    testnet = not args.mainnet
    apply_demo_key_aliases(testnet)

    env_symbol = os.getenv("FUTURES_SYMBOL", SYMBOL)
    env_interval = os.getenv("FUTURES_INTERVAL", INTERVAL)
    if args.symbol != SYMBOL or args.interval != INTERVAL or env_symbol != SYMBOL or env_interval != INTERVAL:
        raise GuardrailViolation("This runner is intentionally locked to FUTURES_SYMBOL=BTCUSDT and FUTURES_INTERVAL=1h")

    dry_run = True
    if args.mode == "demo-supervised" and args.allow_demo_orders and parse_bool(os.getenv("DEMO_TRADING", "")):
        dry_run = False
    if args.mode == "supervised-live" and args.allow_non_dry_run and args.i_understand_live_risk:
        dry_run = False

    config = LiveCandidateConfig(
        mode=args.mode,
        testnet=testnet,
        dry_run=dry_run,
        allocated_capital_usdt=Decimal(str(args.allocated_capital_usdt)) if args.allocated_capital_usdt else decimal_from_env("ALLOCATED_CAPITAL_USDT", "1000"),
        max_notional_usdt=Decimal(str(args.max_notional_usdt)) if args.max_notional_usdt else decimal_from_env("MAX_NOTIONAL_USDT", "25"),
        max_capital_pct_per_trade=Decimal(str(args.max_capital_pct_per_trade)) if args.max_capital_pct_per_trade else decimal_from_env("MAX_CAPITAL_PCT_PER_TRADE", "2.5"),
        max_positions=args.max_positions if args.max_positions is not None else int_from_env("MAX_POSITIONS", "1"),
        max_daily_loss_pct=Decimal(str(args.max_daily_loss_pct)) if args.max_daily_loss_pct else decimal_from_env("MAX_DAILY_LOSS_PCT", "2"),
        max_cumulative_loss_pct=Decimal(str(args.max_cumulative_loss_pct)) if args.max_cumulative_loss_pct else decimal_from_env("MAX_CUMULATIVE_LOSS_PCT", "6"),
        test_order_notional_usdt=Decimal(str(args.test_order_notional_usdt)) if args.test_order_notional_usdt else decimal_from_env("TEST_ORDER_NOTIONAL_USDT", "25"),
        leverage=args.leverage if args.leverage is not None else int_from_env("LEVERAGE", "1"),
        margin_type=(args.margin_type or os.getenv("MARGIN_TYPE", "ISOLATED")).upper(),
        log_file=args.log_file,
    )
    runner = LiveCandidateRunner(config)
    try:
        asyncio.run(runner.run())
    except GuardrailViolation as exc:
        logger.error("Guardrail violation: %s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
