from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from btcusdt_usdm_futures_bot import BinanceFuturesClient, RuntimeConfig
from live_candidate_runner import apply_demo_key_aliases, load_env_file


def mask(value: str) -> str:
    if not value:
        return "EMPTY"
    if len(value) <= 10:
        return "SET"
    return f"{value[:5]}...{value[-5:]}"


def run_check(name: str, fn: Callable[[], object]) -> bool:
    try:
        result = fn()
        print(f"{name}: OK {result}")
        return True
    except Exception as exc:
        print(f"{name}: FAIL {str(exc)[:500]}")
        return False


def main() -> int:
    load_env_file(Path(".env"))
    apply_demo_key_aliases(testnet=True)

    client = BinanceFuturesClient(RuntimeConfig(testnet=True, dry_run=True))
    print("Environment: Binance USD-M Futures Demo/Testnet")
    print(f"Auth type: {client.auth_type}")
    print(f"API key: {mask(os.getenv('BINANCE_FUTURES_API_KEY', ''))}")
    print(f"Private key path: {client.private_key_path or 'EMPTY'}")

    if client.auth_type == "ED25519":
        key = client.load_ed25519_private_key()
        public_key = key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode()
        print("\nDerived public key registered in Binance must match this exactly:")
        print(public_key)

    print("Endpoint checks:")
    ok_public = run_check("exchange_info", lambda: client.exchange_info()["timezone"])
    ok_mark = run_check("mark_price", lambda: client.mark_price("BTCUSDT")["markPrice"])
    ok_account = run_check("account", lambda: list(client.account().keys())[:8])
    ok_position = run_check("position_risk", lambda: client.position_risk("BTCUSDT")[:1])
    ok_listen_key = run_check("listen_key", lambda: "SET" if client.start_listen_key() else "EMPTY")

    if ok_public and ok_mark and not (ok_account and ok_position and ok_listen_key):
        print("\nDiagnosis: public market data works, but Binance rejects private API authentication.")
        print("Action needed in Binance: create/update a Futures Demo API key whose registered public key matches the derived public key above, enable Futures, and verify IP whitelist.")
        return 2

    return 0 if all([ok_public, ok_mark, ok_account, ok_position, ok_listen_key]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
