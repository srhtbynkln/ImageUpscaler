"""Komut satiri: python -m imageupscaler.cli girdi.jpg --cm 100x100 --dpi 312 -o cikti.tif"""
from __future__ import annotations

import argparse
from pathlib import Path

from .core import RECOMMENDED, find_magick, find_upscayl, list_cmyk_profiles, list_models, make_plan, upscale


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Baski icin AI destekli buyutme")
    ap.add_argument("src", nargs="?")
    ap.add_argument("-o", "--out", help="cikti dosyasi (uzanti formati belirler: .tif/.jpg/.png)")
    ap.add_argument("--cm", default="100x100", help="baski boyutu GENxYUK cm (varsayilan 100x100)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--model", default="high-fidelity-4x")
    ap.add_argument("--passes", type=int, default=1, help="AI gecisi sayisi (varsayilan 1)")
    ap.add_argument("--fit", choices=["fit", "crop", "stretch"], default="fit")
    ap.add_argument("--bits", type=int, choices=[8, 16], default=8, help="bit derinligi (16: TIFF/PNG, ImageMagick)")
    ap.add_argument("--color", choices=["rgb", "cmyk"], default="rgb", help="cmyk: TIFF + ICC (ImageMagick)")
    ap.add_argument("--cmyk-icc", help="CMYK ICC profili yolu (yoksa makinede bulunan ilk aday)")
    ap.add_argument("--photoreal", action="store_true", help="uretken doku asamasi (SD karo; venv gerekir)")
    ap.add_argument("--sd-denoise", type=float, default=0.30, help="SD gucu 0.2-0.4 (yuksek=daha az sadik)")
    ap.add_argument("--sd-tile", type=int, default=512)
    ap.add_argument("--sd-steps", type=int, default=20)
    ap.add_argument("--face-fidelity", type=float, default=None, help="CodeFormer (0.9 onerilir; None=kapali)")
    ap.add_argument("--face-blend", type=float, default=0.5)
    ap.add_argument("--grain", type=float, default=None, help="film greni 0.02-0.05")
    ap.add_argument("--mem-limit-gb", type=float, default=12.0, help="bellek bekcisi limiti")
    ap.add_argument("--darken-top", help="ust kenar beyazligini karart: BANT:GUC, orn. 0.02:0.85")
    ap.add_argument("--darken-edges", help="kenar beyazliklarini icteki siyaha uydur: UST:ALT:SOL:SAG:GUC, orn. 0.02:0:0.01:0.01:1")
    ap.add_argument("--face-exposure", type=float, default=None, help="yuz tonu: -0.10 = yuzu %%10 koyulastir (venv gerekir)")
    ap.add_argument("--face-gamma", type=float, default=1.0)
    ap.add_argument("--face-sd", type=float, default=None, help="yuz bolgesine ayri SD gecisi (0.40 onerilir; golgelendirme)")
    ap.add_argument("--recommended", action="store_true", help="onerilen tam zincir: SD 0.30 + gren 0.03 + yuz -0.10 + kenar duzelt + ortadan kirp")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--list-cmyk", action="store_true", help="bulunan CMYK ICC profillerini listele")
    a = ap.parse_args(argv)

    b = find_upscayl()
    if a.list_cmyk:
        print("magick:", find_magick() or "BULUNAMADI")
        print("\n".join(str(p) for p in list_cmyk_profiles()) or "(CMYK profili yok)")
        return 0
    if a.list_models:
        print("upscayl:", b or "BULUNAMADI")
        print("\n".join(list_models(b)) or "(model yok)")
        return 0
    if not a.src:
        ap.error("src gerekli")
    w, h = (float(x) for x in a.cm.lower().split("x"))
    src = Path(a.src)
    out = Path(a.out) if a.out else src.with_name(f"{src.stem}_{a.cm}cm_{a.dpi}dpi.tif")
    plan = make_plan(src, w, h, a.dpi, a.model, a.passes, a.fit, a.bits, a.color,
                     Path(a.cmyk_icc) if a.cmyk_icc else None)
    if a.recommended:
        plan.fit = "crop"
        plan.darken_edges = RECOMMENDED["darken_edges"]; plan.edge_reflect = RECOMMENDED["edge_reflect"]
        plan.photoreal = {"sd_denoise": RECOMMENDED["sd_denoise"], "sd_tile": RECOMMENDED["sd_tile"],
                          "sd_steps": RECOMMENDED["sd_steps"], "grain": RECOMMENDED["grain"],
                          "face_exposure": RECOMMENDED["face_exposure"], "face_gamma": None, "face_sd": RECOMMENDED["face_sd"],
                          "face_fidelity": None, "face_blend": 0.5, "mem_limit_gb": a.mem_limit_gb}
    if a.darken_edges:
        plan.darken_edges = tuple(float(x) for x in a.darken_edges.split(":"))
    if a.darken_top:
        b, st = (float(x) for x in a.darken_top.split(":"))
        plan.darken_top = (b, st)
    if a.photoreal or a.face_fidelity is not None or a.grain is not None or a.face_exposure is not None or a.face_gamma != 1.0 or a.face_sd is not None:
        plan.photoreal = {"face_exposure": a.face_exposure, "face_gamma": a.face_gamma if a.face_gamma != 1.0 else None,
                          "face_sd": a.face_sd,"sd_denoise": a.sd_denoise if a.photoreal else None, "sd_tile": a.sd_tile,
                          "sd_steps": a.sd_steps, "face_fidelity": a.face_fidelity, "face_blend": a.face_blend,
                          "grain": a.grain, "mem_limit_gb": a.mem_limit_gb}
    upscale(src, out, plan, b, fmt=out.suffix.lstrip("."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
