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
    max_notional_usdt: Decimal
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

    async def run(self) -> None:
        self.validate_static_guardrails()
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

    def validate_static_guardrails(self) -> None:
        if self.runtime.symbol != SYMBOL or self.runtime.interval != INTERVAL:
            raise GuardrailViolation("Runner is hard-restricted to BTCUSDT 1h")

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
            if not os.getenv("BINANCE_FUTURES_API_KEY") or not os.getenv("BINANCE_FUTURES_API_SECRET"):
                raise GuardrailViolation("Missing BINANCE_FUTURES_API_KEY or BINANCE_FUTURES_API_SECRET")

        self.event_logger.emit(
            "guardrails_static_passed",
            mode=self.config.mode,
            dry_run=self.config.dry_run,
            testnet=self.config.testnet,
            max_notional=str(self.config.max_notional_usdt),
            leverage=self.config.leverage,
            margin_type=self.config.margin_type,
        )

    async def reconcile_or_abort(self) -> None:
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
            raise GuardrailViolation("No position risk returned by Binance")
        position = positions[0]
        self.state.position_amt = Decimal(str(position.get("positionAmt", "0")))
        self.state.entry_price = Decimal(str(position.get("entryPrice", "0")))
        self.event_logger.emit(
            "position_reconciled",
            position_amt=str(self.state.position_amt),
            entry_price=str(self.state.entry_price),
        )

        if self.config.require_flat_position and not self.state.is_flat:
            raise GuardrailViolation(f"Expected flat position before start, got {self.state.position_amt}")

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
        qty = quantize_down(self.config.test_order_notional_usdt / mark, filters.step_size)
        if qty < filters.min_qty or qty * mark < filters.min_notional:
            raise GuardrailViolation(f"Test order too small after filters: qty={qty}, notional={qty * mark}")

        params = build_market_order_params(SYMBOL, REQUIRED_SIGNAL, qty)
        self.event_logger.emit("test_order_submit", params=sanitize_order_params(params), mark_price=str(mark))
        response = self.client.test_order(params)
        self.event_logger.emit("test_order_accepted", response=response)

    async def start_user_data_stream(self) -> None:
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
            self.event_logger.emit(
                "account_update",
                position_amt=str(self.state.position_amt),
                entry_price=str(self.state.entry_price),
            )
        elif event_type == "ORDER_TRADE_UPDATE":
            self.state.last_order_update = event
            order = event.get("o", {})
            self.event_logger.emit(
                "order_trade_update",
                client_order_id=order.get("c"),
                side=order.get("S"),
                status=order.get("X"),
                execution_type=order.get("x"),
                filled_qty=order.get("z"),
                avg_price=order.get("ap"),
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
        qty = quantize_down(self.config.max_notional_usdt / self.latest_mark_price, filters.step_size)
        if qty < filters.min_qty or qty * self.latest_mark_price < filters.min_notional:
            raise GuardrailViolation(f"Live quantity too small: qty={qty}")

        params = build_market_order_params(SYMBOL, REQUIRED_SIGNAL, qty)
        self.event_logger.emit(f"{event_prefix}_submit_attempt", params=sanitize_order_params(params), candle_time=signal_ts)
        response = self.client.new_order(params)
        self.last_order_signal_ts = signal_ts
        self.event_logger.emit(f"{event_prefix}_response", response=response, candle_time=signal_ts)


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
    parser.add_argument("--max-notional-usdt", default=None)
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
        max_notional_usdt=Decimal(str(args.max_notional_usdt)) if args.max_notional_usdt else decimal_from_env("MAX_NOTIONAL_USDT", "25"),
        test_order_notional_usdt=Decimal(str(args.test_order_notional_usdt)) if args.test_order_notional_usdt else decimal_from_env("TEST_ORDER_NOTIONAL_USDT", "25"),
        leverage=args.leverage if args.leverage is not None else int_from_env("LEVERAGE", "1"),
        margin_type=(args.margin_type or os.getenv("MARGIN_TYPE", "ISOLATED")).upper(),
        log_file=args.log_file,
    )
    runner = LiveCandidateRunner(config)
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
