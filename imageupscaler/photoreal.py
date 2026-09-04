"""Fotogercekci (uretken) asama: yuz icin CodeFormer, tum kare icin SD1.5 + ControlNet Tile.

Restoratif zincir (Real-ESRGAN + Lanczos) kaynaga sadik kalir ama duz/boyanmis gorunur.
Buradaki iki adim Gemini benzeri "fotograf" hissini yerelde uretir; bedeli sadakattir,
o yuzden her ikisinin de fidelity/denoise kolu vardir ve varsayilan olarak KAPALIDIR.

Gereksinim: .venv (torch mps, diffusers, spandrel, facexlib), third_party/CodeFormer,
weights/codeformer.pth + facelib agirliklari, HF'den SD modelleri (ilk kosumda iner).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .guard import DEFAULT_LIMIT_GB, set_mps_cap, start_memory_guard

ROOT = Path(__file__).resolve().parent.parent
CF_DIR = ROOT / "third_party" / "CodeFormer"
W_DIR = ROOT / "weights"

SD_BASE = os.environ.get("UPSCALER_SD_BASE", "SG161222/Realistic_Vision_V5.1_noVAE")
SD_VAE = os.environ.get("UPSCALER_SD_VAE", "stabilityai/sd-vae-ft-mse")
SD_CN_TILE = os.environ.get("UPSCALER_SD_TILE", "lllyasviel/control_v11f1e_sd15_tile")

DEFAULT_PROMPT = ("vintage sepia photograph, elderly gentleman in a dark suit and tie, "
                  "natural skin texture, fine film grain, sharp focus, analog press photo")
DEFAULT_NEG = ("painting, illustration, cartoon, cgi, plastic skin, airbrushed, blurry, "
               "deformed, extra fingers, text, watermark, oversaturated")


def _force_cpu_mps_off() -> None:
    """facelib icerde torch.backends.mps.is_available() ile MPS'e kacar; kapat.
    torch._dynamo bu fonksiyonun __wrapped__ ozniteligini okur -> sahte fonksiyona da ekle."""
    import torch
    orig = torch.backends.mps.is_available
    if getattr(orig, "_imgup_patched", False):
        return
    def off() -> bool:
        return False
    off.__wrapped__ = getattr(orig, "__wrapped__", orig)  # type: ignore[attr-defined]
    off._imgup_patched = True  # type: ignore[attr-defined]
    torch.backends.mps.is_available = off  # type: ignore[assignment]


def _device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ------------------------------------------------------------------ CodeFormer
def _prepare_codeformer_weights() -> None:
    """Indirilen agirliklari CodeFormer'in bekledigi klasorlere baglar."""
    pairs = {
        W_DIR / "codeformer.pth": CF_DIR / "weights" / "CodeFormer" / "codeformer.pth",
        W_DIR / "detection_Resnet50_Final.pth": CF_DIR / "weights" / "facelib" / "detection_Resnet50_Final.pth",
        W_DIR / "parsing_parsenet.pth": CF_DIR / "weights" / "facelib" / "parsing_parsenet.pth",
    }
    for src, dst in pairs.items():
        if src.is_file() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(src, dst)
            except OSError:  # Windows: symlink yetkisi yoksa kopyala
                import shutil
                shutil.copy2(src, dst)
    # CodeFormer deposu basicsr/version.py tasimaz (setup.py uretir); import icin stub yaz
    ver = CF_DIR / "basicsr" / "version.py"
    if (CF_DIR / "basicsr").is_dir() and not ver.is_file():
        ver.write_text("__version__ = '1.3.2'\n__gitsha__ = 'unknown'\nversion_info = (1, 3, 2)\n")


def face_restore(src: Path, dst: Path, fidelity: float = 0.7, device: str | None = None,
                 log: Callable[[str], None] = print) -> int:
    """Yuzleri CodeFormer ile yeniden kurar, ayni boyutta geri yapistirir. Donus: yuz sayisi.

    fidelity 0..1: 1'e yakin = kaynaga sadik (az degisim), 0'a yakin = daha 'guzel' ama uydurma.
    """
    import cv2
    import torch
    from torchvision.transforms.functional import normalize

    start_memory_guard(log=log)
    _prepare_codeformer_weights()
    sys.path.insert(0, str(CF_DIR))
    dev = device or ("cpu" if _device() == "mps" else _device())  # CodeFormer'da MPS'te desteksiz op var; CPU guvenli
    if dev == "cpu":  # facelib icerde kendi get_device()'ini cagirip MPS'e kacar; kapat
        _force_cpu_mps_off()
    from basicsr.utils import img2tensor, tensor2img  # CodeFormer'in kendi basicsr kopyasi
    from basicsr.utils.registry import ARCH_REGISTRY
    from facelib.utils.face_restoration_helper import FaceRestoreHelper

    net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                          connect_list=["32", "64", "128", "256"]).to(dev)
    ck = torch.load(CF_DIR / "weights" / "CodeFormer" / "codeformer.pth", map_location="cpu", weights_only=False)
    net.load_state_dict(ck["params_ema"])
    net.eval()

    helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model="retinaface_resnet50",
                               save_ext="png", use_parse=True, device=dev)
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    helper.read_image(img)
    n = helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
    log(f"CodeFormer: {n} yuz bulundu, fidelity={fidelity}")
    helper.align_warp_face()
    for face in helper.cropped_faces:
        t = img2tensor(face / 255.0, bgr2rgb=True, float32=True)
        normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        t = t.unsqueeze(0).to(dev)
        with torch.no_grad():
            out = net(t, w=fidelity, adain=True)[0]
            rest = tensor2img(out, rgb2bgr=True, min_max=(-1, 1))
        helper.add_restored_face(rest.astype("uint8"), face)
    helper.get_inverse_affine(None)
    res = helper.paste_faces_to_input_image(upsample_img=None, draw_box=False)
    cv2.imwrite(str(dst), res)
    return n


# ------------------------------------------------------------------ SD tile refine
def _blend_mask(w: int, h: int, ov: int) -> np.ndarray:
    """Karo kenarlarinda lineer gecis (0..1)."""
    mx = np.ones(w, dtype=np.float32)
    my = np.ones(h, dtype=np.float32)
    if ov > 0:
        r = np.linspace(0, 1, ov, dtype=np.float32)
        mx[:ov] = r; mx[-ov:] = r[::-1]
        my[:ov] = r; my[-ov:] = r[::-1]
    return my[:, None] * mx[None, :]


def sd_tile_refine(src: Path, dst: Path, denoise: float = 0.35, tile: int = 512, overlap: int = 96,
                   steps: int = 20, guidance: float = 6.0, cn_scale: float = 1.0, seed: int = 1,
                   prompt: str = DEFAULT_PROMPT, negative: str = DEFAULT_NEG, lf_restore: float = 0.0625,
                   log: Callable[[str], None] = print) -> None:
    """Goruntuyu karolara bolup SD1.5 img2img + ControlNet Tile ile yeniden dokular.

    lf_restore: dusuk frekansi (ton/golge) girdiden geri alma; sigma = tile * lf_restore, 0 = kapali.

    denoise 0.2-0.3 = yalniz doku; 0.4+ = biçim de degismeye baslar (sadakat duser).
    """
    import torch
    from diffusers import (AutoencoderKL, ControlNetModel, DPMSolverMultistepScheduler,
                           StableDiffusionControlNetImg2ImgPipeline)

    start_memory_guard(log=log)
    set_mps_cap()
    dev = _device()
    dtype = torch.float16 if dev != "cpu" else torch.float32
    log(f"SD yukleniyor ({dev}, {SD_BASE}) ...")
    cn = ControlNetModel.from_pretrained(SD_CN_TILE, torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(SD_VAE, torch_dtype=dtype)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        SD_BASE, controlnet=cn, vae=vae, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True, final_sigmas_type="sigma_min")
    pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()  # 16 GB birlesik bellekte OOM'a karsi
    pipe.vae.enable_tiling()

    im = Image.open(src).convert("RGB")
    W, H = im.size
    acc = np.zeros((H, W, 3), dtype=np.float32)
    wgt = np.zeros((H, W, 1), dtype=np.float32)
    step = tile - overlap
    xs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
    ys = list(range(0, max(H - tile, 0) + 1, step)) or [0]
    if xs[-1] + tile < W:
        xs.append(W - tile)
    if ys[-1] + tile < H:
        ys.append(H - tile)
    total = len(xs) * len(ys)
    k = 0
    for y in ys:
        for x in xs:
            k += 1
            box = (x, y, min(x + tile, W), min(y + tile, H))
            crop = im.crop(box)
            cw, ch = crop.size
            # SD 8'in kati ister
            cw8, ch8 = cw - cw % 8, ch - ch % 8
            crop8 = crop.resize((cw8, ch8), Image.LANCZOS) if (cw8, ch8) != (cw, ch) else crop
            g = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + y * 10_007 + x)  # karo basina sabit tohum
            out = pipe(prompt=prompt, negative_prompt=negative, image=crop8, control_image=crop8,
                       strength=denoise, num_inference_steps=steps, guidance_scale=guidance,
                       controlnet_conditioning_scale=cn_scale, generator=g).images[0]
            if out.size != (cw, ch):
                out = out.resize((cw, ch), Image.LANCZOS)
            m = _blend_mask(cw, ch, min(overlap, cw // 2, ch // 2))[:, :, None]
            acc[box[1]:box[3], box[0]:box[2]] += np.asarray(out, dtype=np.float32) * m
            wgt[box[1]:box[3], box[0]:box[2]] += m
            log(f"SD karo {k}/{total} ({x},{y})")
            if dev == "mps":
                torch.mps.empty_cache()
    res = acc / np.maximum(wgt, 1e-6)
    if lf_restore > 0:
        # Karo basina ton kaymasi (duz acik fonda "dalga" olarak gorunur): dusuk frekansi GIRDIDEN geri al,
        # SD'nin yalniz yuksek frekans dokusu kalir. sigma = tile * lf_restore (varsayilan tile/16 = 32 px: fon tam geri geldi, doku degismedi).
        from PIL import ImageFilter
        sig = max(4.0, tile * lf_restore)
        lp_in = np.asarray(im.filter(ImageFilter.GaussianBlur(sig)), dtype=np.float32)
        lp_sd = np.asarray(Image.fromarray(np.clip(res, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(sig)),
                           dtype=np.float32)
        res = res - lp_sd + lp_in
        log(f"SD dusuk frekans girdiden geri alindi (sigma {sig:.0f} px)")
    res = np.clip(res, 0, 255).astype(np.uint8)
    Image.fromarray(res).save(dst)


# ------------------------------------------------------------------ yuz tonu
def detect_face_box(im: Image.Image) -> tuple | None:
    """En buyuk yuzu bul: (x, y, w, h) piksel; yoksa None. Once RetinaFace (facelib), yoksa OpenCV Haar."""
    try:
        import torch
        sys.path.insert(0, str(CF_DIR))
        _prepare_codeformer_weights()
        _force_cpu_mps_off()
        from facelib.detection import init_detection_model
        det = init_detection_model("retinaface_resnet50", half=False, device="cpu")
        rgb = np.asarray(im.convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()
        scale = min(1.0, 1024 / max(bgr.shape[:2]))
        if scale < 1:
            import PIL.Image as _I
            small = np.asarray(_I.fromarray(bgr).resize((int(bgr.shape[1] * scale), int(bgr.shape[0] * scale))))
        else:
            small = bgr
        with torch.no_grad():
            boxes = det.detect_faces(small, 0.9)
        if boxes is not None and len(boxes):
            x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))[:4]
            return tuple(int(v / scale) for v in (x1, y1, x2 - x1, y2 - y1))
    except Exception as e:  # noqa: BLE001
        print(f"RetinaFace kullanilamadi ({e}); Haar deneniyor")
    try:
        import cv2
        g = np.asarray(im.convert("L"))
        casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = casc.detectMultiScale(g, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces):
            return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))
    except Exception:  # noqa: BLE001
        pass
    return None


def face_tone(src: Path, dst: Path, exposure: float = -0.10, gamma: float = 1.0,
              expand: float = 1.35, feather: float = 0.35, box: tuple | None = None,
              highlight: float = 0.0, clarity: float = 0.0, knee: float = 0.55,
              log: Callable[[str], None] = print) -> bool:
    """Yuz bolgesinin tonunu ayarlar (eliptik yumusak maske icinde, arka plana dokunmaz).

    exposure: -0.10 = %10 koyulastir (+ aydinlatir). gamma>1 orta tonlari koyular.
    highlight: 0..0.5 parlak cilt alanlarini 'yakmadan' asagi ceker (knee ustu, egim korunur).
    clarity: 0..1 yerel kontrast (genis yaricapli unsharp): goz alti, yanak, burun golgeleri geri gelir.
    Yuz kutusu otomatik (RetinaFace/Haar), box=(x,y,w,h) verilirse tespit atlanir. Donus: yuz bulundu mu.
    """
    from PIL import ImageFilter
    im = Image.open(src).convert("RGB")
    b = box or detect_face_box(im)
    if b is None:
        log("Yuz tonu: yuz bulunamadi, atlandi.")
        im.save(dst); return False
    x, y, w, h = b
    cx, cy = x + w / 2, y + h * 0.55
    rx, ry = w / 2 * expand, h / 2 * expand * 1.15
    W, H = im.size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    mask = np.clip((1 + feather - d) / feather, 0, 1)
    mask = (0.5 * (1 - np.cos(np.pi * mask)))[:, :, None]
    a = np.asarray(im, dtype=np.float32) / 255.0
    adj = a.copy()
    if highlight > 0:                       # luminans uzerinden, renk orani korunur
        L = adj @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        t = np.clip((L - knee) / max(1e-6, 1 - knee), 0, 1); sm = t * t * (3 - 2 * t)
        Ln = L - highlight * sm * L
        ratio = np.where(L > 0.02, Ln / np.maximum(L, 0.02), 1.0)[:, :, None]
        adj = np.clip(adj * ratio, 0, 1)
    if clarity > 0:                         # genis yaricapli yerel kontrast (yalniz luminans)
        rad = max(3, int(w * 0.06))
        L = adj @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        blur = np.asarray(Image.fromarray((L * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(rad)), dtype=np.float32) / 255.0
        Ln = np.clip(L + clarity * (L - blur), 0, 1)
        ratio = np.where(L > 0.02, Ln / np.maximum(L, 0.02), 1.0)[:, :, None]
        adj = np.clip(adj * ratio, 0, 1)
    adj = np.clip(adj * (1 + exposure), 0, 1) ** gamma
    out = a * (1 - mask) + adj * mask
    Image.fromarray(np.clip(out * 255 + 0.5, 0, 255).astype("uint8")).save(dst)
    log(f"Yuz tonu: kutu={b}, exposure={exposure:+.2f}, highlight={highlight}, clarity={clarity}, gamma={gamma}")
    return True


# ------------------------------------------------------------------ grain
def add_grain(src: Path, dst: Path, amount: float = 0.03, size: float = 1.0, seed: int = 1) -> None:
    """Hafif film greni: plastik hissi kirar, baskida gorunmez (amount ~0.02-0.05)."""
    rng = np.random.default_rng(seed)
    im = np.asarray(Image.open(src).convert("RGB"), dtype=np.float32)
    h, w, _ = im.shape
    n = rng.normal(0, 1, (int(h / size), int(w / size))).astype(np.float32)
    if size != 1.0:
        n = np.asarray(Image.fromarray(n).resize((w, h), Image.BILINEAR), dtype=np.float32)
    lum = im.mean(axis=2, keepdims=True) / 255.0
    strength = amount * 255.0 * (0.4 + 0.6 * (1 - np.abs(2 * lum - 1)))  # orta tonlarda daha cok
    out = np.clip(im + n[:, :, None] * strength, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(dst)


# ------------------------------------------------------------------ olcum + orkestra
def fidelity_db(candidate: Path, source: Path) -> float:
    """Adayi kaynak boyutuna indirip PSNR (dB). Yuksek = kaynaga sadik. Kalite olcusu DEGIL."""
    s = Image.open(source).convert("RGB")
    c = Image.open(candidate).convert("RGB").resize(s.size, Image.LANCZOS)
    a = np.asarray(s, dtype=np.float32)
    b = np.asarray(c, dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def blend(a: Path, b: Path, alpha: float, dst: Path) -> None:
    """dst = (1-alpha)*a + alpha*b (ayni boyut)."""
    ia = Image.open(a).convert("RGB")
    ib = Image.open(b).convert("RGB").resize(ia.size, Image.LANCZOS)
    Image.blend(ia, ib, alpha).save(dst)


def photoreal_stage(src_ai: Path, work: Path, source: Path | None = None,
                    face_fidelity: float | None = 0.9, face_blend: float = 0.5,
                    sd_denoise: float | None = 0.3, sd_tile: int = 768, sd_steps: int = 20,
                    grain: float | None = 0.03, prompt: str = DEFAULT_PROMPT,
                    face_exposure: float | None = None, face_gamma: float = 1.0,
                    highlight: float | None = None, contrast: float | None = None,
                    face_highlight: float = 0.0, face_clarity: float = 0.0,
                    face_sd_denoise: float | None = None,
                    log: Callable[[str], None] = print) -> Path:
    """AI 4x ciktisi (src_ai) uzerinde: [CodeFormer yuz (harmanli)] -> [SD karo doku] -> [gren].

    Her adim None ile kapatilir. Donus: son ara dosya (ayni piksel boyutunda).
    source verilirse her adimdan sonra kaynaga sadakat (dB) yazilir.
    """
    work.mkdir(parents=True, exist_ok=True)
    cur = src_ai

    def rep(tag: str, p: Path) -> None:
        if source:
            log(f"  sadakat[{tag}] = {fidelity_db(p, source):.1f} dB")

    rep("ai4x", cur)
    if (highlight is not None and highlight > 0) or (contrast is not None and contrast != 1.0):
        tn = work / "s0_tone.png"
        tone_adjust(cur, tn, highlight=highlight or 0.0, contrast=contrast or 1.0, log=log)
        cur = tn
        rep("tone", cur)
    if face_fidelity is not None:
        f = work / "s1_face.png"
        n = face_restore(cur, f, fidelity=face_fidelity, log=log)
        if n and face_blend < 1.0:
            fb = work / "s1_face_blend.png"
            blend(cur, f, face_blend, fb)
            f = fb
        if n:
            cur = f
        rep("face", cur)
    if sd_denoise is not None:
        s = work / "s2_sd.png"
        sd_tile_refine(cur, s, denoise=sd_denoise, tile=sd_tile, steps=sd_steps, prompt=prompt, log=log)
        cur = s
        rep("sd", cur)
    if face_sd_denoise is not None and face_sd_denoise > 0:
        fs = work / "s22_facesd.png"
        face_sd(cur, fs, denoise=face_sd_denoise, log=log)
        cur = fs
        rep("face_sd", cur)
    if face_exposure is not None or face_gamma != 1.0 or face_highlight > 0 or face_clarity > 0:
        ft = work / "s25_facetone.png"
        face_tone(cur, ft, exposure=face_exposure or 0.0, gamma=face_gamma,
                  highlight=face_highlight, clarity=face_clarity, log=log)
        cur = ft
    if grain is not None and grain > 0:
        g = work / "s3_grain.png"
        add_grain(cur, g, amount=grain)
        cur = g
    return cur


# ------------------------------------------------------------------ ton egrisi (parlak alan kurtarma)
def tone_adjust(src: Path, dst: Path, highlight: float = 0.25, knee: float = 0.55,
                contrast: float = 1.0, shadow: float = 0.0, log: Callable[[str], None] = print) -> None:
    """Parlak alanlari 'yakmadan' asagi ceker (Lightroom 'Highlights -' gibi), renk oranlari korunur.

    highlight: 0..0.5, parlak bolgelerin ne kadar asagi cekilecegi (0.25 = %25). Egim 1'e yakin kalir,
      yani parlak bolgedeki tonlama korunur; yalnizca kaydirilir. knee: etkinin basladigi luminans (0..1).
    contrast: 1 = degismez; 0.9 orta ton etrafinda %10 yumusatir (Gemini benzeri daha yumusak gorunum).
    shadow: 0..0.3 golge kaldirma (isteğe bagli).
    """
    im = Image.open(src).convert("RGB")
    a = np.asarray(im, dtype=np.float32) / 255.0
    L = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    # smoothstep maske: knee'den 1'e
    t = np.clip((L - knee) / max(1e-6, 1 - knee), 0, 1)
    s = t * t * (3 - 2 * t)
    Ln = L - highlight * s * L
    if shadow > 0:
        ts = np.clip(1 - L / max(1e-6, knee), 0, 1)
        Ln = Ln + shadow * (ts * ts) * (1 - L)
    if contrast != 1.0:
        Ln = 0.5 + (Ln - 0.5) * contrast
    ratio = np.where(L > 1e-4, Ln / np.maximum(L, 1e-4), 1.0)[:, :, None]
    out = np.clip(a * ratio, 0, 1)
    Image.fromarray((out * 255 + 0.5).astype("uint8")).save(dst)
    log(f"Ton: highlight={highlight}, knee={knee}, contrast={contrast}, shadow={shadow}")


# ------------------------------------------------------------------ yuz SD (golgelendirme)
FACE_PROMPT = ("vintage sepia photograph, close-up portrait of an elderly gentleman, soft studio lighting, "
               "natural skin shading, subtle shadows under the eyes and cheekbones, realistic skin texture, "
               "fine film grain, sharp focus")


def face_sd(src: Path, dst: Path, denoise: float = 0.45, size: int = 768, expand: float = 1.7,
            feather: float = 0.30, steps: int = 24, guidance: float = 6.0, cn_scale: float = 0.9,
            seed: int = 1, prompt: str = FACE_PROMPT, negative: str = DEFAULT_NEG,
            box: tuple | None = None, log: Callable[[str], None] = print) -> bool:
    """Yuz bolgesini ayri bir SD img2img + ControlNet Tile gecisiyle yeniden isikla (golgeler, cilt).

    Yuz kutusu buyutulup kare kesilir, `size` px'e getirilir, `denoise` gucuyle kosar, eliptik yumusak
    maskeyle geri yapistirilir. Genel SD gecisinden BAGIMSIZ ve daha guclu; sadakati dusurur.
    """
    import torch
    from diffusers import (AutoencoderKL, ControlNetModel, DPMSolverMultistepScheduler,
                           StableDiffusionControlNetImg2ImgPipeline)
    start_memory_guard(log=log); set_mps_cap()
    im = Image.open(src).convert("RGB")
    b = box or detect_face_box(im)
    if b is None:
        log("Yuz SD: yuz bulunamadi, atlandi."); im.save(dst); return False
    x, y, w, h = b
    W, H = im.size
    cx, cy = x + w / 2, y + h * 0.5
    side = int(max(w, h) * expand)
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(W, x0 + side)), int(min(H, y0 + side))
    crop = im.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    work = crop.resize((size, size), Image.LANCZOS)
    dev = _device(); dtype = torch.float16 if dev != "cpu" else torch.float32
    log(f"Yuz SD: kutu={b}, kesit {cw}x{ch} -> {size}px, denoise={denoise}")
    cn = ControlNetModel.from_pretrained(SD_CN_TILE, torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(SD_VAE, torch_dtype=dtype)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        SD_BASE, controlnet=cn, vae=vae, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++",
                                                             use_karras_sigmas=True, final_sigmas_type="sigma_min")
    pipe.to(dev); pipe.set_progress_bar_config(disable=True); pipe.enable_attention_slicing()
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = pipe(prompt=prompt, negative_prompt=negative, image=work, control_image=work, strength=denoise,
               num_inference_steps=steps, guidance_scale=guidance, controlnet_conditioning_scale=cn_scale,
               generator=g).images[0]
    out = out.resize((cw, ch), Image.LANCZOS)
    # eliptik yumusak maske (yuz kutusuna gore, kesit koordinatinda)
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    fx, fy = cx - x0, cy - y0
    rx, ry = w / 2 * 1.25, h / 2 * 1.35
    d = np.sqrt(((xx - fx) / rx) ** 2 + ((yy - fy) / ry) ** 2)
    m = np.clip((1 + feather - d) / feather, 0, 1); m = (0.5 * (1 - np.cos(np.pi * m)))[:, :, None]
    a = np.asarray(crop, dtype=np.float32); o = np.asarray(out, dtype=np.float32)
    blended = Image.fromarray(np.clip(a * (1 - m) + o * m, 0, 255).astype("uint8"))
    im.paste(blended, (x0, y0))
    im.save(dst)
    if dev == "mps":
        torch.mps.empty_cache()
    return True
