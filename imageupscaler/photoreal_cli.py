"""Fotogercekci asama alt sureci (venv python ile kosar; GUI/CLI bunu subprocess olarak cagirir).

python -m imageupscaler.photoreal_cli --in ai4x.png --out-dir work --source kaynak.jpg \
    [--face-fidelity 0.9 --face-blend 0.5] [--sd-denoise 0.3 --sd-tile 512 --sd-steps 20] [--grain 0.03]
Son satir:  SON <yol>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source", help="orijinal kaynak (sadakat dB icin)")
    ap.add_argument("--face-fidelity", type=float, default=None)
    ap.add_argument("--face-blend", type=float, default=0.5)
    ap.add_argument("--sd-denoise", type=float, default=None)
    ap.add_argument("--sd-tile", type=int, default=512)
    ap.add_argument("--sd-steps", type=int, default=20)
    ap.add_argument("--grain", type=float, default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--face-exposure", type=float, default=None, help="yuz tonu: -0.10 = %%10 koyu")
    ap.add_argument("--face-gamma", type=float, default=1.0)
    ap.add_argument("--highlight", type=float, default=None, help="GLOBAL parlak alan kurtarma 0..0.5 (onerilmez)")
    ap.add_argument("--face-highlight", type=float, default=0.0, help="yuz ici parlak alan kurtarma 0..0.5")
    ap.add_argument("--face-clarity", type=float, default=0.0, help="yuz ici yerel kontrast 0..1")
    ap.add_argument("--face-sd", type=float, default=None, help="yuz bolgesine ayri SD gecisi gucu (0.40 onerilir)")
    ap.add_argument("--contrast", type=float, default=None, help="1=degismez, 0.9 yumusak")
    a = ap.parse_args(argv)
    from .photoreal import DEFAULT_PROMPT, photoreal_stage

    def log(s: str) -> None:
        print(s, flush=True)

    p = photoreal_stage(Path(a.src), Path(a.out_dir), Path(a.source) if a.source else None,
                        face_fidelity=a.face_fidelity, face_blend=a.face_blend,
                        sd_denoise=a.sd_denoise, sd_tile=a.sd_tile, sd_steps=a.sd_steps,
                        grain=a.grain, prompt=a.prompt or DEFAULT_PROMPT,
                        face_exposure=a.face_exposure, face_gamma=a.face_gamma,
                        highlight=a.highlight, contrast=a.contrast,
                        face_highlight=a.face_highlight, face_clarity=a.face_clarity,
                        face_sd_denoise=a.face_sd, log=log)
    print(f"SON {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
