import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Deque, Optional

import websockets


# =========================
# CONFIGURACIÓN
# =========================
SYMBOLS = ["btcusdt", "ethusdt"]
INTERVAL = "1m"

EMA_FAST_PERIOD = 9
EMA_SLOW_PERIOD = 21
RSI_PERIOD = 14
MAX_CLOSES = 300

RECONNECT_DELAY_SECONDS = 5
HEARTBEAT_SECONDS = 30

WS_BASE_URL = "wss://stream.binance.com:9443"
STREAMS = "/".join(f"{symbol}@kline_{INTERVAL}" for symbol in SYMBOLS)
WS_URL = f"{WS_BASE_URL}/stream?streams={STREAMS}"


# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("binance_ws_bot")


# =========================
# MODELOS
# =========================
@dataclass
class Candle:
    symbol: str
    open_time: int
    close_time: int
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal
    is_closed: bool

    @property
    def close_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.close_time / 1000, tz=timezone.utc)


@dataclass
class Signal:
    symbol: str
    side: str
    price: Decimal
    ema_fast: Decimal
    ema_slow: Decimal
    rsi: Decimal
    event_time: datetime


# =========================
# INDICADORES
# =========================
class EMA:
    """
    EMA incremental:
    - En la primera etapa acumula valores hasta completar el período.
    - Luego usa la fórmula incremental estándar.
    """

    def __init__(self, period: int):
        self.period = period
        self.multiplier = Decimal("2") / Decimal(period + 1)
        self._seed_values: list[Decimal] = []
        self.current: Optional[Decimal] = None

    def update(self, price: Decimal) -> Optional[Decimal]:
        if self.current is None:
            self._seed_values.append(price)
            if len(self._seed_values) < self.period:
                return None

            # Seed inicial usando SMA
            self.current = sum(self._seed_values) / Decimal(self.period)
            return self.current

        self.current = (price - self.current) * self.multiplier + self.current
        return self.current


class RSI:
    """
    RSI incremental estilo Wilder.
    Necesita un primer bloque de cambios para inicializar avg_gain / avg_loss.
    """

    def __init__(self, period: int):
        self.period = period
        self.prev_close: Optional[Decimal] = None
        self._seed_gains: list[Decimal] = []
        self._seed_losses: list[Decimal] = []
        self.avg_gain: Optional[Decimal] = None
        self.avg_loss: Optional[Decimal] = None
        self.current: Optional[Decimal] = None

    def update(self, close_price: Decimal) -> Optional[Decimal]:
        if self.prev_close is None:
            self.prev_close = close_price
            return None

        change = close_price - self.prev_close
        gain = max(change, Decimal("0"))
        loss = max(-change, Decimal("0"))

        if self.avg_gain is None or self.avg_loss is None:
            self._seed_gains.append(gain)
            self._seed_losses.append(loss)
            self.prev_close = close_price

            if len(self._seed_gains) < self.period:
                return None

            self.avg_gain = sum(self._seed_gains) / Decimal(self.period)
            self.avg_loss = sum(self._seed_losses) / Decimal(self.period)
            self.current = self._calculate_rsi(self.avg_gain, self.avg_loss)
            return self.current

        self.avg_gain = ((self.avg_gain * Decimal(self.period - 1)) + gain) / Decimal(self.period)
        self.avg_loss = ((self.avg_loss * Decimal(self.period - 1)) + loss) / Decimal(self.period)
        self.prev_close = close_price
        self.current = self._calculate_rsi(self.avg_gain, self.avg_loss)
        return self.current

    @staticmethod
    def _calculate_rsi(avg_gain: Decimal, avg_loss: Decimal) -> Decimal:
        if avg_loss == 0:
            return Decimal("100")
        rs = avg_gain / avg_loss
        return Decimal("100") - (Decimal("100") / (Decimal("1") + rs))


# =========================
# ESTADO POR SÍMBOLO
# =========================
@dataclass
class SymbolState:
    symbol: str
    closes: Deque[Decimal] = field(default_factory=lambda: deque(maxlen=MAX_CLOSES))
    ema_fast: EMA = field(default_factory=lambda: EMA(EMA_FAST_PERIOD))
    ema_slow: EMA = field(default_factory=lambda: EMA(EMA_SLOW_PERIOD))
    rsi: RSI = field(default_factory=lambda: RSI(RSI_PERIOD))

    last_ema_fast: Optional[Decimal] = None
    last_ema_slow: Optional[Decimal] = None
    current_ema_fast: Optional[Decimal] = None
    current_ema_slow: Optional[Decimal] = None
    current_rsi: Optional[Decimal] = None

    candles_processed: int = 0
    last_candle_time: Optional[datetime] = None

    def on_closed_candle(self, candle: Candle) -> Optional[Signal]:
        close_price = candle.close_price
        self.closes.append(close_price)

        # Guardamos valores previos para detectar cruce
        prev_fast = self.current_ema_fast
        prev_slow = self.current_ema_slow

        new_fast = self.ema_fast.update(close_price)
        new_slow = self.ema_slow.update(close_price)
        new_rsi = self.rsi.update(close_price)

        self.last_ema_fast = prev_fast
        self.last_ema_slow = prev_slow
        self.current_ema_fast = new_fast
        self.current_ema_slow = new_slow
        self.current_rsi = new_rsi
        self.candles_processed += 1
        self.last_candle_time = candle.close_datetime

        if not self.is_ready():
            return None

        return self._build_signal(candle)

    def is_ready(self) -> bool:
        return (
            self.last_ema_fast is not None
            and self.last_ema_slow is not None
            and self.current_ema_fast is not None
            and self.current_ema_slow is not None
            and self.current_rsi is not None
        )

    def _build_signal(self, candle: Candle) -> Optional[Signal]:
        assert self.last_ema_fast is not None
        assert self.last_ema_slow is not None
        assert self.current_ema_fast is not None
        assert self.current_ema_slow is not None
        assert self.current_rsi is not None

        crossed_up = self.last_ema_fast <= self.last_ema_slow and self.current_ema_fast > self.current_ema_slow
        crossed_down = self.last_ema_fast >= self.last_ema_slow and self.current_ema_fast < self.current_ema_slow

        # Filtro RSI simple para evitar señales demasiado “ciegas”
        if crossed_up and self.current_rsi > Decimal("55"):
            return Signal(
                symbol=self.symbol,
                side="LONG",
                price=candle.close_price,
                ema_fast=self.current_ema_fast,
                ema_slow=self.current_ema_slow,
                rsi=self.current_rsi,
                event_time=candle.close_datetime,
            )

        if crossed_down and self.current_rsi < Decimal("45"):
            return Signal(
                symbol=self.symbol,
                side="SHORT",
                price=candle.close_price,
                ema_fast=self.current_ema_fast,
                ema_slow=self.current_ema_slow,
                rsi=self.current_rsi,
                event_time=candle.close_datetime,
            )

        return None


# =========================
# PARSER
# =========================
def parse_decimal(value: str) -> Decimal:
    try:
        return Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"No se pudo convertir a Decimal: {value}") from exc


def parse_candle_message(raw_message: str) -> Optional[Candle]:
    """
    Binance combined stream:
    {
      "stream": "btcusdt@kline_1m",
      "data": {
        "e": "kline",
        "E": 123456789,
        "s": "BTCUSDT",
        "k": {...}
      }
    }
    """
    payload = json.loads(raw_message)

    data = payload.get("data")
    if not data:
        return None

    if data.get("e") != "kline":
        return None

    k = data.get("k")
    if not k:
        return None

    return Candle(
        symbol=data["s"].lower(),
        open_time=int(k["t"]),
        close_time=int(k["T"]),
        open_price=parse_decimal(k["o"]),
        high_price=parse_decimal(k["h"]),
        low_price=parse_decimal(k["l"]),
        close_price=parse_decimal(k["c"]),
        volume=parse_decimal(k["v"]),
        is_closed=bool(k["x"]),
    )


# =========================
# MOTOR PRINCIPAL
# =========================
class MarketStreamApp:
    def __init__(self, ws_url: str, symbols: list[str]):
        self.ws_url = ws_url
        self.symbol_states = {symbol: SymbolState(symbol=symbol) for symbol in symbols}
        self._last_message_at: Optional[datetime] = None

    async def run_forever(self) -> None:
        while True:
            try:
                logger.info("Conectando a %s", self.ws_url)
                async with websockets.connect(self.ws_url) as websocket:
                    logger.info("Conexión WebSocket establecida")
                    self._last_message_at = datetime.now(tz=timezone.utc)

                    consumer_task = asyncio.create_task(self._consume_messages(websocket))
                    heartbeat_task = asyncio.create_task(self._heartbeat())

                    done, pending = await asyncio.wait(
                        {consumer_task, heartbeat_task},
                        return_when=asyncio.FIRST_EXCEPTION,
                    )

                    for task in pending:
                        task.cancel()

                    for task in done:
                        exc = task.exception()
                        if exc:
                            raise exc

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Se perdió la conexión o ocurrió un error: %s", exc)
                logger.info("Reintentando en %s segundos...", RECONNECT_DELAY_SECONDS)
                await asyncio.sleep(RECONNECT_DELAY_SECONDS)

    async def _consume_messages(self, websocket) -> None:
        async for raw_message in websocket:
            self._last_message_at = datetime.now(tz=timezone.utc)

            candle = parse_candle_message(raw_message)
            if candle is None:
                continue

            # Procesamos solo velas cerradas para evitar ruido intrabar
            if not candle.is_closed:
                continue

            state = self.symbol_states.get(candle.symbol)
            if state is None:
                logger.warning("Símbolo no esperado: %s", candle.symbol)
                continue

            signal = state.on_closed_candle(candle)

            logger.info(
                "[%s] candle cerrada | close=%s | ema_fast=%s | ema_slow=%s | rsi=%s | candles=%s",
                candle.symbol.upper(),
                candle.close_price,
                _fmt_decimal(state.current_ema_fast),
                _fmt_decimal(state.current_ema_slow),
                _fmt_decimal(state.current_rsi),
                state.candles_processed,
            )

            if signal:
                self._handle_signal(signal)

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)

            now = datetime.now(tz=timezone.utc)
            if self._last_message_at is None:
                logger.warning("Aún no se recibió ningún mensaje")
                continue

            silence_seconds = int((now - self._last_message_at).total_seconds())
            logger.info("Heartbeat | segundos desde último mensaje: %s", silence_seconds)

            for symbol, state in self.symbol_states.items():
                logger.info(
                    "Estado %s | candles=%s | última vela=%s | close_count=%s",
                    symbol.upper(),
                    state.candles_processed,
                    state.last_candle_time.isoformat() if state.last_candle_time else "N/A",
                    len(state.closes),
                )

    def _handle_signal(self, signal: Signal) -> None:
        logger.warning(
            "SEÑAL %s | %s | price=%s | ema_fast=%s | ema_slow=%s | rsi=%s | time=%s",
            signal.side,
            signal.symbol.upper(),
            signal.price,
            _fmt_decimal(signal.ema_fast),
            _fmt_decimal(signal.ema_slow),
            _fmt_decimal(signal.rsi),
            signal.event_time.isoformat(),
        )


def _fmt_decimal(value: Optional[Decimal], places: str = "0.0000") -> str:
    if value is None:
        return "N/A"
    return str(value.quantize(Decimal(places)))


# =========================
# ENTRY POINT
# =========================
async def main() -> None:
    logger.info("Iniciando aplicación de streaming...")
    logger.info("Símbolos: %s", ", ".join(symbol.upper() for symbol in SYMBOLS))
    logger.info("Intervalo: %s", INTERVAL)

    app = MarketStreamApp(ws_url=WS_URL, symbols=SYMBOLS)
    await app.run_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Aplicación detenida manualmente.")