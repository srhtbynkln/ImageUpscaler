"""Bellek bekcisi: surec RSS'i limiti asarsa isi durdurur (makineyi sisirmez).

Kullanim: start_memory_guard(12.0)  -> arka planda izler; asilirsa os._exit(3) ile HEMEN cikar.
Ayrica torch/MPS icin set_mps_cap(limit_gb) Metal ayirmalarina sert tavan koyar (OOM hatasi verir,
swap'e kacmaz). Limit ortam degiskeniyle de verilir: UPSCALER_MEM_LIMIT_GB (varsayilan 12).
"""
from __future__ import annotations

import os
import sys
import threading
import time

DEFAULT_LIMIT_GB = float(os.environ.get("UPSCALER_MEM_LIMIT_GB", "12"))


def rss_gb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 2**30


def start_memory_guard(limit_gb: float = DEFAULT_LIMIT_GB, interval: float = 0.5,
                       log=print) -> threading.Thread:
    def watch() -> None:
        peak = 0.0
        while True:
            r = rss_gb()
            peak = max(peak, r)
            if r > limit_gb:
                log(f"BELLEK BEKCISI: RSS {r:.1f} GB > limit {limit_gb:.1f} GB; is durduruldu (tepe {peak:.1f} GB).")
                sys.stdout.flush()
                os._exit(3)
            time.sleep(interval)
    t = threading.Thread(target=watch, daemon=True, name="memguard")
    t.start()
    return t


def set_mps_cap(limit_gb: float = DEFAULT_LIMIT_GB) -> None:
    """Metal ayirmalarina tavan: limit / toplam RAM orani. Asilirsa torch OOM hatasi verir."""
    try:
        import psutil
        import torch
        if torch.backends.mps.is_available():
            total = psutil.virtual_memory().total / 2**30
            torch.mps.set_per_process_memory_fraction(min(0.95, limit_gb / total))
    except Exception:  # noqa: BLE001
        pass
