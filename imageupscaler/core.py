"""Cekirdek: Upscayl (Real-ESRGAN ncnn) ile AI gecisi + Pillow Lanczos ile tam piksel.

Terimler
- AI gecisi: Real-ESRGAN tabanli modelin goruntuyu TEK seferde 4x buyutmesi.
  Model her piksel icin komsuluga bakip "burada ne olmaliydi" tahmini yapar
  (JPEG bloklarini temizler, kenarlari toparlar). Tahmin olduğu icin her gecis
  bir miktar uydurma doku ekleyebilir; iki gecis ust uste (16x) bu uydurmayi
  katlar. Varsayilan 1 gecis, kalan oran Lanczos (yorumsuz, klasik) ile alinir.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from PIL import Image, ImageCms

Image.MAX_IMAGE_PIXELS = None  # 12k x 12k gibi boyutlarda Pillow'un bomba korumasini kapat

INCH_CM = 2.54

# ---------------------------------------------------------------- iptal
CANCEL = threading.Event()
_PROCS: list = []          # calisan alt surecler (Popen); cancel() hepsini oldurur
_PROCS_LOCK = threading.Lock()


class Cancelled(RuntimeError):
    pass


def cancel() -> None:
    """Calisan isi durdur: bayragi kaldir, alt surecleri oldur."""
    CANCEL.set()
    with _PROCS_LOCK:
        for pr in list(_PROCS):
            try:
                pr.terminate()
            except Exception:  # noqa: BLE001
                pass


def reset_cancel() -> None:
    CANCEL.clear()


def _check_cancel() -> None:
    if CANCEL.is_set():
        raise Cancelled("Durduruldu")


class Stalled(RuntimeError):
    """Alt surec ilerleme yazmadan takili kaldi (GPU baslatma kilidi)."""


def _run_tracked(cmd, log: Optional[Callable[[str], None]] = None, heartbeat: float = 30.0,
                 stall_after: float = 120.0, retries: int = 2,
                 **kw) -> subprocess.CompletedProcess:
    """subprocess.run gibi, ama surec kaydedilir (cancel() oldurur) ve her `heartbeat` sn'de yasam isareti yazar.

    Takilma bekcisi: surec `stall_after` sn boyunca stdout/stderr'e TEK BAYT yazmadiysa oldurulur ve `retries`
    kez yeniden baslatilir; sonra Stalled firlatilir. (upscayl-bin GPU'da calisirken CPU harcamaz ama her karoda
    yuzde yazar; 04.09.2026'da GPU baslatmada 61 dk sessiz asili kaldi. CPU suresi olcut DEGIL: 3,5 dk'lik saglikli
    kosu da 1 sn CPU harciyor.)
    """
    import threading
    import time

    for attempt in range(retries + 1):
        _check_cancel()
        pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              bufsize=1, **kw)
        with _PROCS_LOCK:
            _PROCS.append(pr)
        bufs = {"out": [], "err": []}
        last = [time.time()]

        def reader(stream, key):
            try:
                while True:
                    ch = stream.read(1)
                    if not ch:
                        break
                    bufs[key].append(ch)
                    last[0] = time.time()
            except Exception:  # noqa: BLE001
                pass

        th = [threading.Thread(target=reader, args=(pr.stdout, "out"), daemon=True),
              threading.Thread(target=reader, args=(pr.stderr, "err"), daemon=True)]
        for t in th:
            t.start()
        t0 = time.time()
        stalled = False
        try:
            while pr.poll() is None:
                try:
                    pr.wait(timeout=heartbeat)
                except subprocess.TimeoutExpired:
                    el = time.time() - t0
                    quiet = time.time() - last[0]
                    if log:
                        log(f"  ... calisiyor ({int(el)} sn, son cikti {int(quiet)} sn once), {Path(cmd[0]).name}")
                    if quiet >= stall_after:
                        stalled = True
                        pr.kill()
                        pr.wait()
                        break
            for t in th:
                t.join(timeout=5)
        finally:
            with _PROCS_LOCK:
                if pr in _PROCS:
                    _PROCS.remove(pr)
        _check_cancel()
        out, err = "".join(bufs["out"]), "".join(bufs["err"])
        if not stalled:
            return subprocess.CompletedProcess(cmd, pr.returncode, out, err)
        if log:
            log(f"  TAKILDI: {int(stall_after)} sn boyunca cikti yok; surec olduruldu"
                + (f", yeniden deneme {attempt + 1}/{retries}" if attempt < retries else ""))
    raise Stalled(f"{Path(cmd[0]).name} {retries + 1} denemede de takildi (GPU baslatma?)")


# Onerilen ayar (04.09.2026, 600 px sepya portre -> 1 m baski ile dogrulandi)
RECOMMENDED = {
    "model": "high-fidelity-4x", "ai_passes": 1, "fit": "crop", "fmt": "tiff",
    "sd_denoise": 0.30, "sd_tile": 512, "sd_steps": 20, "grain": 0.03,
    "face_exposure": -0.10, "face_sd": 0.0, "mem_limit_gb": 12.0,
    "darken_edges": (0.012, 0.004, 0.010, 0.008, 1.0), "edge_reflect": (0.008, 0.002, 0.005, 0.004),
}
AI_SCALE = 4  # Upscayl modelleri 4x


# ---------------------------------------------------------------- Upscayl bul
def _candidates() -> list[Path]:
    env = os.environ.get("UPSCAYL_BIN")
    c: list[Path] = [Path(env)] if env else []
    s = platform.system()
    if s == "Darwin":
        c += [Path("/Applications/Upscayl.app/Contents/Resources/bin/upscayl-bin"),
              Path.home() / "Applications/Upscayl.app/Contents/Resources/bin/upscayl-bin"]
    elif s == "Windows":
        la = os.environ.get("LOCALAPPDATA", "")
        pf = os.environ.get("PROGRAMFILES", r"C:\Program Files")
        c += [Path(la) / r"Programs\Upscayl\resources\bin\upscayl-bin.exe",
              Path(pf) / r"Upscayl\resources\bin\upscayl-bin.exe"]
    else:
        c += [Path("/opt/Upscayl/resources/bin/upscayl-bin"),
              Path("/usr/lib/upscayl/resources/bin/upscayl-bin")]
    w = shutil.which("upscayl-bin") or shutil.which("realesrgan-ncnn-vulkan")
    if w:
        c.append(Path(w))
    return c


def find_upscayl() -> Optional[Path]:
    for p in _candidates():
        if p.is_file():
            return p
    return None


def models_dir(bin_path: Path) -> Optional[Path]:
    env = os.environ.get("UPSCAYL_MODELS")
    if env and Path(env).is_dir():
        return Path(env)
    for d in (bin_path.parent.parent / "models", bin_path.parent / "models"):
        if d.is_dir():
            return d
    return None


def list_models(bin_path: Optional[Path]) -> list[str]:
    if not bin_path:
        return []
    d = models_dir(bin_path)
    if not d:
        return []
    return sorted(p.stem for p in d.glob("*.param"))


def venv_python() -> Optional[Path]:
    """Fotogercekci asama icin torch'lu venv (repo/.venv). Yoksa None."""
    root = Path(__file__).resolve().parent.parent
    for p in (root / ".venv" / "bin" / "python", root / ".venv" / "Scripts" / "python.exe"):
        if p.is_file():
            return p
    return None


def run_photoreal(src: Path, work: Path, source: Optional[Path], opts: dict,
                  log: Callable[[str], None] = print) -> Path:
    """photoreal_cli'yi venv python ile ayri surecte kosar; bellek bekcisi o surecte calisir."""
    py = venv_python()
    if not py:
        raise RuntimeError("Fotogercekci asama icin repo/.venv yok. Kurulum: README 'Fotogercekci asama'.")
    root = Path(__file__).resolve().parent.parent
    cmd = [str(py), "-m", "imageupscaler.photoreal_cli", "--in", str(src), "--out-dir", str(work)]
    if source:
        cmd += ["--source", str(source)]
    for k in ("face_fidelity", "face_blend", "sd_denoise", "sd_tile", "sd_steps", "grain", "prompt",
              "face_exposure", "face_gamma", "highlight", "contrast", "face_highlight", "face_clarity", "face_sd"):
        v = opts.get(k)
        if v is not None:
            cmd += ["--" + k.replace("_", "-"), str(v)]
    env = dict(os.environ)
    if opts.get("mem_limit_gb"):
        env["UPSCALER_MEM_LIMIT_GB"] = str(opts["mem_limit_gb"])
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    log("> " + " ".join(cmd))
    _check_cancel()
    proc = subprocess.Popen(cmd, cwd=str(root), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    with _PROCS_LOCK:
        _PROCS.append(proc)
    last = None
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if not line or "arning" in line or "eprecat" in line or "it/s" in line:
            continue
        log("  " + line)
        if line.startswith("SON "):
            last = Path(line[4:].strip())
    rc = proc.wait()
    with _PROCS_LOCK:
        if proc in _PROCS:
            _PROCS.remove(proc)
    _check_cancel()
    if rc == 3:
        raise RuntimeError("Fotogercekci asama bellek limitini asti ve durduruldu (bekci). Karo boyutunu kucult.")
    if rc != 0 or last is None or not last.is_file():
        raise RuntimeError(f"Fotogercekci asama basarisiz (kod {rc}).")
    return last


def find_magick() -> Optional[str]:
    """16 bit ve CMYK icin ImageMagick 7 (`magick`). Yoksa None."""
    env = os.environ.get("MAGICK_BIN")
    if env and Path(env).is_file():
        return env
    return shutil.which("magick")


_CMYK_DIRS = {
    "Darwin": ["/System/Library/ColorSync/Profiles", "/Library/ColorSync/Profiles",
               "/Library/Application Support/Adobe/Color/Profiles/Recommended",
               "/Library/Application Support/Adobe/Color/Profiles", "~/Library/ColorSync/Profiles"],
    "Windows": [r"C:\Windows\System32\spool\drivers\color",
                r"C:\Program Files (x86)\Common Files\Adobe\Color\Profiles\Recommended",
                r"C:\Program Files (x86)\Common Files\Adobe\Color\Profiles",
                r"C:\Program Files\Common Files\Adobe\Color\Profiles"],
    "Linux": ["/usr/share/color/icc", "~/.color/icc", "~/.local/share/icc"],
}
_CMYK_HINT = ("fogra", "swop", "coated", "cmyk", "gracol", "japan", "eci")


def list_cmyk_profiles() -> list[Path]:
    """Makinedeki CMYK ICC adaylari (ad ipucuna gore). FOGRA39/ISO Coated one cikar."""
    env = os.environ.get("UPSCALER_CMYK_ICC")
    found: list[Path] = [Path(env)] if env and Path(env).is_file() else []
    for d in _CMYK_DIRS.get(platform.system(), []):
        dp = Path(os.path.expanduser(d))
        if dp.is_dir():
            for f in list(dp.glob("*.icc")) + list(dp.glob("*.icm")):
                if any(h in f.name.lower() for h in _CMYK_HINT) and f not in found:
                    found.append(f)
    pri = lambda p: (0 if "fogra39" in p.name.lower() or "iso coated" in p.name.lower()
                     else 1 if "coated" in p.name.lower() else 2 if "generic" not in p.name.lower() else 3)
    return sorted(found, key=lambda p: (pri(p), p.name.lower()))


# ---------------------------------------------------------------- plan
@dataclass
class Plan:
    src_w: int
    src_h: int
    target_w: int
    target_h: int
    dpi: int
    ai_passes: int
    model: str
    fit: str                      # "fit" | "crop" | "stretch"
    bit_depth: int = 8            # 8 | 16 (16 yalniz TIFF/PNG, ImageMagick gerekir)
    color: str = "rgb"            # "rgb" | "cmyk" (CMYK yalniz TIFF, ImageMagick gerekir)
    cmyk_profile: Optional[Path] = None
    photoreal: Optional[dict] = None
    darken_top: Optional[tuple] = None  # (bant orani 0..0.2, guc 0..1): ust kenardaki tarama beyazligini karart
    darken_edges: Optional[tuple] = None  # (ust, alt, sol, sag, guc): kenar beyazliklarini icteki siyaha uydur
    edge_reflect: Optional[tuple] = None  # (ust, alt, sol, sag): dis serit doldurma genisligi; None = varsayilan   # {"face_fidelity","face_blend","sd_denoise","sd_tile","sd_steps","grain","mem_limit_gb"}
    ai_w: int = 0                 # AI gecisleri sonrasi boyut
    ai_h: int = 0
    linear_scale: float = 0.0     # hedef / kaynak (kenar orani)
    notes: list[str] = field(default_factory=list)


def target_pixels(width_cm: float, height_cm: float, dpi: int) -> tuple[int, int]:
    return (round(width_cm / INCH_CM * dpi), round(height_cm / INCH_CM * dpi))


def make_plan(src: Path, width_cm: float, height_cm: float, dpi: int,
              model: str, ai_passes: int = 1, fit: str = "fit",
              bit_depth: int = 8, color: str = "rgb",
              cmyk_profile: Optional[Path] = None) -> Plan:
    with Image.open(src) as im:
        sw, sh = im.size
        src_bits = 16 if im.mode in ("I;16", "I;16B", "I;16L", "I") else 8
    tw, th = target_pixels(width_cm, height_cm, dpi)
    p = Plan(sw, sh, tw, th, dpi, ai_passes, model, fit, bit_depth, color, cmyk_profile)
    if bit_depth == 16 and src_bits == 8:
        p.notes.append("NOT: kaynak 8 bit; 16 bit cikti yeni bilgi eklemez, yalniz matbaa hattinin "
                       "istedigi konteynerdir (gecisler 16 bit hesaplanir, tonlama bantlasmaz).")
    if color == "cmyk":
        if cmyk_profile is None:
            c = list_cmyk_profiles()
            p.cmyk_profile = c[0] if c else None
        if p.cmyk_profile is None:
            raise ValueError("CMYK icin ICC profili yok. FOGRA39/ISO Coated v2'yi eci.org'dan indirip "
                             "UPSCALER_CMYK_ICC ile ver veya GUI'de sec.")
        p.notes.append(f"CMYK profili: {p.cmyk_profile.name} (matbaanin istedigi profil degilse degistir).")
    if (bit_depth == 16 or color == "cmyk") and not find_magick():
        raise ValueError("16 bit / CMYK icin ImageMagick 7 gerekir (macOS: brew install imagemagick; "
                         "Windows: imagemagick.org). Bulunamadi.")
    if fit == "fit":  # en-boy korunur, hedef kutuya sigacak en buyuk boyut
        s = min(tw / sw, th / sh)
        p.target_w, p.target_h = round(sw * s), round(sh * s)
    p.linear_scale = max(p.target_w / sw, p.target_h / sh)
    p.ai_w, p.ai_h = sw * AI_SCALE ** ai_passes, sh * AI_SCALE ** ai_passes
    eff = min(sw / (width_cm / INCH_CM), sh / (height_cm / INCH_CM))
    p.notes.append(f"Kaynak {sw}x{sh}px; bu boyutta gercek cozunurluk ~{eff:.0f} dpi. "
                   f"Hedef {p.target_w}x{p.target_h}px @ {dpi} dpi (x{p.linear_scale:.2f}).")
    if ai_passes >= 2:
        p.notes.append("UYARI: 2+ AI gecisi uydurma doku (catlak/damar) uretebilir; 1 gecis onerilir.")
    if p.linear_scale > 8:
        p.notes.append("UYARI: >8x buyutmede kaybolan detay geri gelmez; sonucu 100% yakinlikta kontrol et.")
    if fit == "stretch" and abs((tw / th) - (sw / sh)) > 0.01:
        p.notes.append("UYARI: stretch en-boy oranini bozar.")
    return p


# ---------------------------------------------------------------- calistir
def run_ai_pass(bin_path: Path, src: Path, dst: Path, model: str, tile: int = 0,
                log: Callable[[str], None] = print) -> None:
    md = models_dir(bin_path)
    cmd = [str(bin_path), "-i", str(src), "-o", str(dst), "-s", str(AI_SCALE),
           "-n", model, "-f", "png", "-t", str(tile)]
    if md:
        cmd += ["-m", str(md)]
    log("> " + " ".join(cmd))
    r = _run_tracked(cmd, log=log)
    if r.returncode != 0 or not dst.is_file():
        raise RuntimeError(f"upscayl-bin basarisiz ({r.returncode}; -9 = bellek yetmedi, girdi cok buyuk):\n{r.stdout}\n{r.stderr}")


def find_border(im: Image.Image, thresh: float = 30.0, pad: int = 2) -> tuple[int, int, int, int]:
    """Cerceve (siyah/koyu kenar) disindaki icerigin kutusunu bul: (x0, y0, x1, y1) kaynak piksel.

    Satir/sutun ortalamasi `thresh` altinda kaldigi surece kenardan iceri yurur; `pad` piksel guvenlik payi
    iceri dogru eklenir (cerceve kalintisi kalmasin). Beyaz cerceve icin thresh'i yuksek ve `invert` mantigi
    gerekmez: parlak cerceve icin (255 - L) uzerinden cagir.
    """
    import numpy as np
    g = np.asarray(im.convert("L"), dtype=np.float32)
    H, W = g.shape
    rows, cols = g.mean(axis=1), g.mean(axis=0)

    def run(v):
        a = 0
        while a < len(v) - 1 and v[a] < thresh:
            a += 1
        b = len(v)
        while b > a + 1 and v[b - 1] < thresh:
            b -= 1
        return a, b

    t, b = run(rows); l, r = run(cols)
    t, l = min(t + pad, H - 1), min(l + pad, W - 1)
    b, r = max(b - pad, t + 1), max(r - pad, l + 1)
    return l, t, r, b


def darken_edges(im: Image.Image, bands=(0.02, 0.0, 0.01, 0.01), strength: float = 1.0,
                 black: Optional[tuple] = None, edge: float = 0.0, texture: bool = True,
                 reflect=(0.005, 0.002, 0.004, 0.003)) -> Image.Image:
    """Tarama kenari beyazliklarini o bolgenin IC tonuna uydurur; gren/doku korunur, gorunur gecis olmaz.

    bands=(ust, alt, sol, sag): ton duzeltme bandi (yukseklik/genislik orani).
    reflect=(ust, alt, sol, sag): en dis serit, hemen icindeki dokunun AYNASIYLA doldurulur
      (tarama cizgisi/siyah kenar tamamen gider, doku surekli kalir). 0 = kapali.
    Yontem: goruntu = dusuk frekans (ton) + yuksek frekans (gren). Bantta ton, icteki referanstan
    parlaksa referansa cekilir (asla aydinlatmaz), kosinus rampa; gren geri eklenir ama parlak yonlu
    gren ic bolgenin tipik genligine kirpilir (ince parlak cizgi sizamaz).
    edge>0: en dis serit ayrica `black` tonuna cekilir (varsayilan kapali; ic ton daha dogaldir).
    """
    import numpy as np
    from PIL import ImageFilter
    a = np.asarray(im.convert("RGB"), dtype=np.float32)
    h, w, _ = a.shape
    # 1) yansitma: dis serit = icteki dokunun aynasi
    # dis serit = icteki dokunun KAYDIRILMIS kopyasi (ayna simetrik motif birakir), dikiste kisa capraz gecis
    rt, rb, rl, rr = (max(0, int(round(f * (h if i < 2 else w)))) for i, f in enumerate(reflect))

    def xfade(n: int) -> np.ndarray:
        k = max(2, min(n // 2, int(min(h, w) * 0.003)))
        return k, np.linspace(0, 1, k, dtype=np.float32)

    if rt:
        src = a[rt:2 * rt].copy(); k, t = xfade(rt)
        src[-k:] = src[-k:] * (1 - t[:, None, None]) + a[rt - k:rt] * t[:, None, None]
        a[:rt] = src
    if rb:
        src = a[h - 2 * rb:h - rb].copy(); k, t = xfade(rb)
        src[:k] = a[h - rb:h - rb + k] * (1 - t[:, None, None]) + src[:k] * t[:, None, None]
        a[h - rb:] = src
    if rl:
        src = a[:, rl:2 * rl].copy(); k, t = xfade(rl)
        src[:, -k:] = src[:, -k:] * (1 - t[None, :, None]) + a[:, rl - k:rl] * t[None, :, None]
        a[:, :rl] = src
    if rr:
        src = a[:, w - 2 * rr:w - rr].copy(); k, t = xfade(rr)
        src[:, :k] = a[:, w - rr:w - rr + k] * (1 - t[None, :, None]) + src[:, :k] * t[None, :, None]
        a[:, w - rr:] = src
    rgb = Image.fromarray(np.clip(a, 0, 255).astype("uint8"))
    sig = max(2, int(min(h, w) * 0.003))
    low = np.asarray(rgb.filter(ImageFilter.GaussianBlur(sig)), dtype=np.float32)
    high = a - low if texture else np.zeros_like(a)
    if black is None:
        lum = a.mean(axis=2); dark = a[lum < 60]
        black = tuple(np.median(dark, axis=0)) if len(dark) else (0.0, 0.0, 0.0)
    blk = np.array(black, dtype=np.float32)

    def cos_ramp(n_total: int, reverse: bool, plateau: int = 0) -> np.ndarray:
        """Kenardan `plateau` boyunca 1, sonra kosinusle 0'a iner."""
        n_ramp = max(1, n_total - plateau)
        x = np.arange(n_ramp, dtype=np.float32) / max(1, n_ramp - 1)
        r = np.concatenate([np.ones(plateau, dtype=np.float32), 0.5 * (1 + np.cos(np.pi * x))])[:n_total]
        return r[::-1] if reverse else r

    def smooth_ref(ref: np.ndarray) -> np.ndarray:
        return np.asarray(Image.fromarray(np.clip(ref, 0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(sig * 3)), dtype=np.float32)

    top, bottom, left, right = (max(0, int(round(f * (h if i < 2 else w)))) for i, f in enumerate(bands))
    out_low = low.copy()
    hi_w = np.ones((h, w, 1), dtype=np.float32)   # bantta parlak gren kirpma agirligi (ramp)

    def fix(n: int, axis: int, from_start: bool) -> None:
        L = h if axis == 0 else w
        nt = min(L // 2, int(n * 1.5))
        if from_start:
            ref = (low[n:2 * n] if axis == 0 else low[:, n:2 * n]).mean(axis=axis, keepdims=True)
            sl = slice(0, nt); ramp = cos_ramp(nt, False, plateau=int(n * 0.6))
        else:
            ref = (low[L - 2 * n:L - n] if axis == 0 else low[:, L - 2 * n:L - n]).mean(axis=axis, keepdims=True)
            sl = slice(L - nt, L); ramp = cos_ramp(nt, True, plateau=int(n * 0.6))
        ref = smooth_ref(ref)
        r = (ramp[:, None, None] if axis == 0 else ramp[None, :, None]) * strength
        if axis == 0:
            band = out_low[sl]; out_low[sl] = band * (1 - r) + np.minimum(band, ref) * r
            hi_w[sl] = np.minimum(hi_w[sl], 1 - r)
            if edge > 0:
                ne = max(1, int(round(edge * L))); er = cos_ramp(ne, not from_start)[:, None, None]
                s2 = slice(0, ne) if from_start else slice(L - ne, L)
                out_low[s2] = out_low[s2] * (1 - er) + blk * er
        else:
            band = out_low[:, sl]; out_low[:, sl] = band * (1 - r) + np.minimum(band, ref) * r
            hi_w[:, sl] = np.minimum(hi_w[:, sl], 1 - r)
            if edge > 0:
                ne = max(1, int(round(edge * L))); er = cos_ramp(ne, not from_start)[None, :, None]
                s2 = slice(0, ne) if from_start else slice(L - ne, L)
                out_low[:, s2] = out_low[:, s2] * (1 - er) + blk * er

    if top: fix(top, 0, True)
    if bottom: fix(bottom, 0, False)
    if left: fix(left, 1, True)
    if right: fix(right, 1, False)
    # gren: bant icinde parlak yonlu bileseni ic bolgenin tipik genligine kirp (ramp agirlikli)
    m = max(top, bottom, left, right, 1)
    core = high[m:h - m, m:w - m] if (h > 3 * m and w > 3 * m) else high
    cap = float(np.percentile(core, 99.0))
    high_c = np.minimum(high, cap)
    high_out = high * hi_w + high_c * (1 - hi_w)
    out = out_low + high_out
    return Image.fromarray(np.clip(out, 0, 255).astype("uint8"))


def darken_top(im: Image.Image, band: float = 0.02, strength: float = 0.85, thresh: float = 0.30) -> Image.Image:
    """Ust kenar bandindaki PARLAK pikselleri koyulastirir (tarama kenari beyazligi).

    band: yukseklik orani (0.02 = ustten %2); strength: en ustte karartma orani; koyu pikseller dokunulmaz.
    Rampa ustte 1, bant altinda 0; parlaklik esigi thresh uzerindeki pikseller yumusak secilir.
    """
    import numpy as np
    a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    h = a.shape[0]
    n = max(1, int(round(h * band)))
    lum = a[:n].mean(axis=2)
    bright = np.clip((lum - thresh) / max(1e-6, 0.5 - thresh), 0, 1)       # 0 koyu .. 1 parlak
    ramp = (1.0 - np.arange(n, dtype=np.float32) / n)[:, None]              # 1 ustte .. 0 bant sonu
    k = 1.0 - strength * ramp * bright                                       # carpan
    a[:n] *= k[:, :, None]
    return Image.fromarray(np.clip(a * 255, 0, 255).astype("uint8"))


def _fit_resize(im: Image.Image, w: int, h: int, fit: str) -> Image.Image:
    if fit == "crop":
        s = max(w / im.width, h / im.height)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        l, t = (im.width - w) // 2, (im.height - h) // 2
        return im.crop((l, t, l + w, t + h))
    return im.resize((w, h), Image.LANCZOS)  # fit: plan zaten oranli; stretch: zorla


def save_print(im: Image.Image, dst: Path, dpi: int, fmt: str, icc: Optional[bytes],
               jpeg_quality: int = 95) -> None:
    fmt = fmt.lower()
    kw: dict = {"dpi": (dpi, dpi)}
    if icc:
        kw["icc_profile"] = icc
    if fmt in ("tif", "tiff"):
        kw["compression"] = "tiff_lzw"
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        im.save(dst, "TIFF", **kw)
    elif fmt in ("jpg", "jpeg"):
        if im.mode != "RGB":
            im = im.convert("RGB")
        im.save(dst, "JPEG", quality=jpeg_quality, subsampling=0, optimize=True, **kw)
    elif fmt == "png":
        im.save(dst, "PNG", **kw)
    else:
        raise ValueError(f"bilinmeyen format: {fmt}")


def upscale(src: Path, dst: Path, plan: Plan, bin_path: Optional[Path],
            fmt: str = "tiff", log: Callable[[str], None] = print) -> Path:
    src, dst = Path(src), Path(dst)
    reset_cancel()
    for n in plan.notes:
        log(n)
    with Image.open(src) as im0:
        icc = im0.info.get("icc_profile")
        base = im0.convert("RGB")
    SD_MAX = 3200        # fotogercekci (SD) asamasinin calisacagi en buyuk kenar (px); ustu saatler surer
    SD_PRE_MIN = 1500    # kaynak bu kadar buyukse SD, AI gecisinden ONCE kosar (kaynak zaten detayli)
    with tempfile.TemporaryDirectory(prefix="imgup_") as td:
        cur = Path(td) / "p0.png"
        # a) hedef orani icin kirpma ONCE (AI'nin kirpilacak alani islememesi icin)
        if plan.fit == "crop":
            tw, th = plan.target_w, plan.target_h
            s_ = max(tw / base.width, th / base.height)
            cw, ch = min(base.width, int(round(tw / s_))), min(base.height, int(round(th / s_)))
            if (cw, ch) != base.size:
                l, t = (base.width - cw) // 2, (base.height - ch) // 2
                base = base.crop((l, t, l + cw, t + ch))
                log(f"Once kirpildi (hedef orani): {cw}x{ch}px")
        # b) kaynak x 4^gecis hedefi asiyorsa once kucult: AI ciktisi tam hedefe otursun (bellek + sure)
        passes = plan.ai_passes if bin_path else 0
        if passes:
            k = AI_SCALE ** passes
            need_w, need_h = plan.target_w / k, plan.target_h / k
            if base.width > need_w * 1.02 or base.height > need_h * 1.02:
                sc = max(need_w / base.width, need_h / base.height)
                nw, nh = max(1, int(round(base.width * sc))), max(1, int(round(base.height * sc)))
                base = base.resize((nw, nh), Image.LANCZOS)
                log(f"Kaynak hedef/{k}'e kucultuldu: {nw}x{nh}px (AI {k}x sonrasi ~{nw * k}x{nh * k})")
        base.save(cur, "PNG")
        fid_src = cur   # sadakat olcumu kirpilmis/kucultulmus tabana gore (orijinal src ile boyut uyusmaz)
        if plan.ai_passes and not bin_path:
            log("UYARI: Upscayl bulunamadi; yalniz Lanczos uygulanacak (AI gecisi yok).")
        # c) fotogercekci: buyuk kaynakta AI'dan ONCE (SD 3200 px ustunde saatler surer), gren daima AI'dan sonra
        pr = dict(plan.photoreal) if plan.photoreal else None
        grain_later = None
        if pr:
            grain_later = pr.pop("grain", None)
            pre = max(base.size) >= SD_PRE_MIN
            if pre and any(pr.get(k) for k in ("sd_denoise", "face_exposure", "face_gamma", "highlight", "contrast", "face_highlight", "face_clarity", "face_sd")):
                if max(base.size) > SD_MAX and pr.get("sd_denoise") is not None:
                    log(f"NOT: kaynak {max(base.size)} px > {SD_MAX}; SD dokusu ATLANDI (saatler surer). Kaynak zaten detayli.")
                    pr["sd_denoise"] = None
                if any(pr.get(k) for k in ("sd_denoise", "face_exposure", "highlight", "contrast", "face_highlight", "face_clarity", "face_sd")):
                    log("Fotogercekci asama AI gecisinden ONCE kosuyor (kaynak buyuk) ...")
                    cur = run_photoreal(cur, Path(td) / "photoreal_pre", fid_src, pr, log=log)
                pr = None   # sonrasinda yalniz gren
        for i in range(passes):
            nxt = Path(td) / f"p{i + 1}.png"
            log(f"AI gecisi {i + 1}/{passes} ({plan.model}) ...")
            run_ai_pass(bin_path, cur, nxt, plan.model, log=log)
            cur = nxt
        _check_cancel()
        if pr:   # kucuk kaynak: SD/yuz AI'dan sonra
            with Image.open(cur) as chk_im:
                big = max(chk_im.size)
            if big > SD_MAX and pr.get("sd_denoise") is not None:
                log(f"NOT: AI ciktisi {big} px > {SD_MAX}; SD dokusu ATLANDI (saatler surer).")
                pr["sd_denoise"] = None
            if any(pr.get(k) for k in ("sd_denoise", "face_exposure", "highlight", "contrast", "face_highlight", "face_clarity", "face_sd")):
                log("Fotogercekci asama (ton/yuz/SD doku) ...")
                cur = run_photoreal(cur, Path(td) / "photoreal", fid_src, pr, log=log)
        if grain_later:
            log(f"Gren {grain_later} ...")
            cur = run_photoreal(cur, Path(td) / "grain", None,
                                {"grain": grain_later, "mem_limit_gb": (plan.photoreal or {}).get("mem_limit_gb")}, log=log)
        if plan.darken_edges:
            t, b_, l, r_, st = plan.darken_edges
            log(f"Kenar karartma: ust %{t*100:.1f} alt %{b_*100:.1f} sol %{l*100:.1f} sag %{r_*100:.1f}, guc {st}")
            de = Path(td) / "edges.png"
            with Image.open(cur) as im:
                kw = {"reflect": plan.edge_reflect} if plan.edge_reflect else {}
                darken_edges(im, (t, b_, l, r_), st, **kw).save(de, "PNG")
            cur = de
        if plan.darken_top:
            b, st = plan.darken_top
            log(f"Ust kenar karartma: bant %{b*100:.1f}, guc {st}")
            dt = Path(td) / "darken.png"
            with Image.open(cur) as im:
                darken_top(im, b, st).save(dt, "PNG")
            cur = dt
        _check_cancel()
        log(f"Lanczos ile {plan.target_w}x{plan.target_h} px'e getiriliyor ...")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if plan.bit_depth == 16 or plan.color == "cmyk":
            _magick_finish(cur, dst, plan, fmt, icc, Path(td), log)
        else:
            with Image.open(cur) as im:
                out = _fit_resize(im.convert("RGB"), plan.target_w, plan.target_h, plan.fit)
                save_print(out, dst, plan.dpi, fmt, icc)
    with Image.open(dst) as chk:  # geri oku: sessiz kayit yok
        log(f"Yazildi: {dst} {chk.size[0]}x{chk.size[1]} mod={chk.mode} dpi={chk.info.get('dpi')}")
    return dst


def _magick_finish(cur: Path, dst: Path, plan: Plan, fmt: str, icc: Optional[bytes],
                   td: Path, log: Callable[[str], None]) -> None:
    """ImageMagick (Q16): 16 bit Lanczos, sRGB->CMYK ICC donusumu, dpi etiketi, LZW."""
    fmt = fmt.lower()
    if plan.color == "cmyk" and fmt not in ("tif", "tiff"):
        log("UYARI: CMYK yalniz TIFF'e yazilir; format TIFF yapildi.")
        fmt, dst = "tiff", dst.with_suffix(".tif")
    if plan.bit_depth == 16 and fmt in ("jpg", "jpeg"):
        log("UYARI: JPEG 16 bit tasimaz; 8 bit yazilacak.")
    src_icc = td / "src.icc"
    src_icc.write_bytes(icc if icc else ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    w, h = plan.target_w, plan.target_h
    cmd = [find_magick(), str(cur), "-depth", "16", "-filter", "Lanczos"]
    if plan.fit == "crop":
        cmd += ["-resize", f"{w}x{h}^", "-gravity", "center", "-extent", f"{w}x{h}"]
    else:
        cmd += ["-resize", f"{w}x{h}!"]
    cmd += ["-profile", str(src_icc)]          # kaynak renk uzayini ata (yoksa sRGB)
    if plan.color == "cmyk":
        cmd += ["-profile", str(plan.cmyk_profile)]  # ikinci -profile = DONUSUM
    cmd += ["-units", "PixelsPerInch", "-density", str(plan.dpi),
            "-depth", "16" if plan.bit_depth == 16 and fmt != "jpg" else "8"]
    if fmt in ("tif", "tiff"):
        cmd += ["-compress", "LZW"]
    elif fmt in ("jpg", "jpeg"):
        cmd += ["-quality", "95", "-sampling-factor", "1x1"]
    cmd += ["-limit", "memory", "4GiB", "-limit", "map", "8GiB", str(dst)]
    log("> " + " ".join(cmd))
    r = _run_tracked(cmd, log=log)
    if r.returncode != 0 or not dst.is_file():
        raise RuntimeError(f"magick basarisiz ({r.returncode}):\n{r.stderr}")
