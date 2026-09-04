"""Tkinter GUI: python -m imageupscaler.gui

Ozellikler: dosya sec / surukle-birak, onizleme uzerinde fare ile kirpma (oran kilidi), tum ayarlar,
onerilen varsayilanlar (core.RECOMMENDED), ayri surecte fotogercekci asama, bellek bekcisi.
"""
from __future__ import annotations

import queue
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .core import (INCH_CM, RECOMMENDED, Cancelled, cancel, find_border, find_magick, find_upscayl,
                   list_cmyk_profiles, list_models, make_plan, target_pixels, upscale)

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _BaseTk = TkinterDnD.Tk
    HAS_DND = True
except Exception:  # noqa: BLE001
    _BaseTk = tk.Tk
    HAS_DND = False

Image.MAX_IMAGE_PIXELS = None
PRESETS = {"Ozel": None, "100x100 cm": (100, 100), "70x100 cm (B1)": (70, 100),
           "50x70 cm": (50, 70), "A3 29.7x42": (29.7, 42), "A4 21x29.7": (21, 29.7)}
PREVIEW = 620      # onizleme kenari (px)
HANDLE = 7         # kose tutamaci yaricapi
LOUPE = 220        # buyutec penceresi (px)
LOUPE_ZOOM = 6     # buyutec: kaynak pikseli kac kat


class App(_BaseTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ImageUpscaler")
        self.bin = find_upscayl()
        self.q: queue.Queue[str] = queue.Queue()
        self.img: Image.Image | None = None      # kaynak (tam)
        self.disp: ImageTk.PhotoImage | None = None
        self.scale = 1.0                          # onizleme / kaynak
        self.off = (0, 0)                         # onizlemenin canvas icindeki konumu
        self.crop: list[float] | None = None      # kaynak piksel [x0,y0,x1,y1]
        self.quad: list[list[float]] | None = None  # 4 kose (nw, ne, se, sw) kaynak piksel; perspektif modu
        self.img_orig: Image.Image | None = None  # yuklenen orijinal (onay sonrasi img = islenmis)
        self.confirmed = False                    # kirpim/perspektif onaylandi mi
        self._drag = None
        self._running = False
        self.active_corner: str | None = None     # ok tuslariyla oynatilacak kose
        self._loupe_img = None
        self._build()
        self.after(200, self._drain)

    # ------------------------------------------------------------ arayuz
    def _build(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.grid(sticky="nsew")
        self.columnconfigure(0, weight=1); self.rowconfigure(0, weight=1)
        left = ttk.Frame(root); left.grid(row=0, column=0, sticky="nsew")
        right = ttk.Frame(root); right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        root.columnconfigure(0, weight=1); root.rowconfigure(0, weight=1)
        f = left
        r = 0
        ttk.Label(f, text="Girdi:").grid(row=r, column=0, sticky="w")
        self.src = tk.StringVar()
        e = ttk.Entry(f, textvariable=self.src, width=44); e.grid(row=r, column=1, columnspan=2, sticky="ew")
        ttk.Button(f, text="Sec / Yukle", command=self._pick_src).grid(row=r, column=3)
        e.bind("<Return>", lambda *_: self.load_image(self.src.get()))
        r += 1
        ttk.Label(f, text="Cikti klasoru:").grid(row=r, column=0, sticky="w")
        self.outdir = tk.StringVar(value=str(Path.home() / "Desktop"))
        ttk.Entry(f, textvariable=self.outdir, width=44).grid(row=r, column=1, columnspan=2, sticky="ew")
        ttk.Button(f, text="Sec", command=self._pick_out).grid(row=r, column=3)
        r += 1
        ttk.Label(f, text="Baski boyutu:").grid(row=r, column=0, sticky="w")
        self.preset = tk.StringVar(value="100x100 cm")
        cb = ttk.Combobox(f, textvariable=self.preset, values=list(PRESETS), state="readonly", width=14)
        cb.grid(row=r, column=1, sticky="w"); cb.bind("<<ComboboxSelected>>", self._preset)
        self.wcm, self.hcm = tk.DoubleVar(value=100), tk.DoubleVar(value=100)
        sub = ttk.Frame(f); sub.grid(row=r, column=2, columnspan=2, sticky="w")
        ttk.Entry(sub, textvariable=self.wcm, width=6).pack(side="left")
        ttk.Label(sub, text=" x ").pack(side="left")
        ttk.Entry(sub, textvariable=self.hcm, width=6).pack(side="left")
        ttk.Label(sub, text=" cm").pack(side="left")
        r += 1
        ttk.Label(f, text="Hedef:").grid(row=r, column=0, sticky="w")
        ht = ttk.Frame(f); ht.grid(row=r, column=1, columnspan=3, sticky="w")
        self.tmode = tk.StringVar(value="cm")
        ttk.Radiobutton(ht, text="cm + dpi", value="cm", variable=self.tmode, command=self._pixels).pack(side="left")
        ttk.Radiobutton(ht, text="olcek", value="scale", variable=self.tmode, command=self._pixels).pack(side="left")
        self.scale_sel = tk.StringVar(value="4x")
        cbs = ttk.Combobox(ht, textvariable=self.scale_sel, values=["2x", "4x", "8x", "16x"], state="readonly", width=5)
        cbs.pack(side="left"); cbs.bind("<<ComboboxSelected>>", lambda *_: self._pixels())
        ttk.Label(ht, text=" (olcek: kaynak px x olcek; dpi yalniz etiket)").pack(side="left")
        r += 1
        ttk.Label(f, text="DPI:").grid(row=r, column=0, sticky="w")
        self.dpi = tk.IntVar(value=312)
        ttk.Spinbox(f, from_=72, to=2400, textvariable=self.dpi, width=7).grid(row=r, column=1, sticky="w")
        self.pix = ttk.Label(f, text=""); self.pix.grid(row=r, column=2, columnspan=2, sticky="w")
        for v in (self.wcm, self.hcm, self.dpi):
            v.trace_add("write", lambda *_: self._pixels())
        r += 1
        ttk.Label(f, text="Model:").grid(row=r, column=0, sticky="w")
        ms = list_models(self.bin)
        self.model = tk.StringVar(value=RECOMMENDED["model"] if RECOMMENDED["model"] in ms else (ms[0] if ms else ""))
        ttk.Combobox(f, textvariable=self.model, values=ms, state="readonly", width=22).grid(row=r, column=1, columnspan=2, sticky="w")
        r += 1
        ttk.Label(f, text="AI gecisi:").grid(row=r, column=0, sticky="w")
        self.passes = tk.IntVar(value=RECOMMENDED["ai_passes"])
        ttk.Spinbox(f, from_=0, to=3, textvariable=self.passes, width=4).grid(row=r, column=1, sticky="w")
        ttk.Label(f, text="0=yalniz Lanczos, 1=onerilen, 2+=uydurma doku riski").grid(row=r, column=2, columnspan=2, sticky="w")
        r += 1
        ttk.Label(f, text="Sigdirma:").grid(row=r, column=0, sticky="w")
        self.fit = tk.StringVar(value=RECOMMENDED["fit"])
        fr = ttk.Frame(f); fr.grid(row=r, column=1, columnspan=3, sticky="w")
        for k, t in (("fit", "oran koru"), ("crop", "ortadan kirp"), ("stretch", "esnet")):
            ttk.Radiobutton(fr, text=t, value=k, variable=self.fit).pack(side="left")
        r += 1
        ttk.Label(f, text="Format:").grid(row=r, column=0, sticky="w")
        self.fmt = tk.StringVar(value=RECOMMENDED["fmt"])
        fr2 = ttk.Frame(f); fr2.grid(row=r, column=1, columnspan=3, sticky="w")
        for k, t in (("tiff", "TIFF (LZW, baski)"), ("jpg", "JPEG q95"), ("png", "PNG")):
            ttk.Radiobutton(fr2, text=t, value=k, variable=self.fmt).pack(side="left")
        r += 1
        ttk.Label(f, text="Renk / bit:").grid(row=r, column=0, sticky="w")
        fr3 = ttk.Frame(f); fr3.grid(row=r, column=1, columnspan=3, sticky="w")
        self.color, self.bits = tk.StringVar(value="rgb"), tk.IntVar(value=8)
        ttk.Radiobutton(fr3, text="RGB", value="rgb", variable=self.color).pack(side="left")
        ttk.Radiobutton(fr3, text="CMYK (TIFF+ICC)", value="cmyk", variable=self.color).pack(side="left")
        ttk.Label(fr3, text="   ").pack(side="left")
        ttk.Radiobutton(fr3, text="8 bit", value=8, variable=self.bits).pack(side="left")
        ttk.Radiobutton(fr3, text="16 bit", value=16, variable=self.bits).pack(side="left")
        r += 1
        ttk.Label(f, text="CMYK ICC:").grid(row=r, column=0, sticky="w")
        self.profiles = list_cmyk_profiles()
        self.icc = tk.StringVar(value=str(self.profiles[0]) if self.profiles else "")
        ttk.Combobox(f, textvariable=self.icc, values=[str(p) for p in self.profiles], width=42).grid(row=r, column=1, columnspan=2, sticky="ew")
        ttk.Button(f, text="Sec", command=self._pick_icc).grid(row=r, column=3)
        r += 1
        ttk.Label(f, text="Fotogercekci:").grid(row=r, column=0, sticky="w")
        fr4 = ttk.Frame(f); fr4.grid(row=r, column=1, columnspan=3, sticky="w")
        self.pr_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(fr4, text="SD doku", variable=self.pr_on).pack(side="left")
        ttk.Label(fr4, text=" guc").pack(side="left")
        self.pr_denoise = tk.DoubleVar(value=RECOMMENDED["sd_denoise"])
        ttk.Spinbox(fr4, from_=0.1, to=0.6, increment=0.05, textvariable=self.pr_denoise, width=5).pack(side="left")
        ttk.Label(fr4, text=" gren").pack(side="left")
        self.pr_grain = tk.DoubleVar(value=RECOMMENDED["grain"])
        ttk.Spinbox(fr4, from_=0.0, to=0.1, increment=0.01, textvariable=self.pr_grain, width=5).pack(side="left")
        ttk.Label(fr4, text=" yuz SD").pack(side="left")
        self.fsd = tk.DoubleVar(value=RECOMMENDED["face_sd"])
        ttk.Spinbox(fr4, from_=0.0, to=0.7, increment=0.05, textvariable=self.fsd, width=5).pack(side="left")
        ttk.Label(fr4, text=" yuz ton").pack(side="left")
        self.ftone = tk.DoubleVar(value=RECOMMENDED["face_exposure"])
        ttk.Spinbox(fr4, from_=-0.5, to=0.5, increment=0.05, textvariable=self.ftone, width=5).pack(side="left")
        r += 1
        ttk.Label(f, text="Kenar / bellek:").grid(row=r, column=0, sticky="w")
        fr5 = ttk.Frame(f); fr5.grid(row=r, column=1, columnspan=3, sticky="w")
        self.dedge = tk.BooleanVar(value=True)
        ttk.Checkbutton(fr5, text="kenar duzelt", variable=self.dedge).pack(side="left")
        ttk.Label(fr5, text="   bellek limiti GB").pack(side="left")
        self.mem = tk.DoubleVar(value=RECOMMENDED["mem_limit_gb"])
        ttk.Spinbox(fr5, from_=4, to=64, increment=1, textvariable=self.mem, width=5).pack(side="left")
        r += 1
        bf = ttk.Frame(f); bf.grid(row=r, column=1, columnspan=2, sticky="w", pady=6)
        self.btn = ttk.Button(bf, text="Buyut", command=self._run); self.btn.pack(side="left")
        self.btn_stop = ttk.Button(bf, text="Durdur", command=self._stop, state="disabled"); self.btn_stop.pack(side="left", padx=4)
        self.btn_restart = ttk.Button(bf, text="Yeniden baslat", command=self._restart, state="disabled"); self.btn_restart.pack(side="left")
        self.pb = ttk.Progressbar(f, mode="indeterminate"); self.pb.grid(row=r, column=3, sticky="ew")
        r += 1
        self.log = tk.Text(f, height=11, width=70, state="disabled")
        self.log.grid(row=r, column=0, columnspan=4, sticky="nsew")
        f.rowconfigure(r, weight=1); f.columnconfigure(2, weight=1)

        # ---- sag: onizleme + kirpma
        ttk.Label(right, text="Onizleme / kirpma  (surukle: tasi, kose: boyut, dis: yeni)").grid(row=0, column=0, columnspan=4, sticky="w")
        self.cv = tk.Canvas(right, width=PREVIEW, height=PREVIEW, bg="#333", highlightthickness=1, highlightbackground="#888")
        self.cv.grid(row=1, column=0, columnspan=4)
        self.cv.bind("<ButtonPress-1>", self._cv_press)
        self.cv.bind("<B1-Motion>", self._cv_drag)
        self.cv.bind("<ButtonRelease-1>", self._cv_release)
        self.cv.bind("<Motion>", self._cv_motion)
        self.cv.bind("<Leave>", lambda e: self.cv.delete("loupe"))
        for k in ("<Left>", "<Right>", "<Up>", "<Down>", "<Shift-Left>", "<Shift-Right>", "<Shift-Up>", "<Shift-Down>"):
            self.bind(k, self._nudge)
        self.lock = tk.BooleanVar(value=True)
        ttk.Checkbutton(right, text="Oran kilidi (baski oraninda)", variable=self.lock, command=self._crop_lock_changed).grid(row=2, column=0, sticky="w")
        ttk.Button(right, text="Tumu", command=self._crop_all).grid(row=2, column=1, sticky="w")
        ttk.Button(right, text="Kenar bul", command=self._crop_auto_border).grid(row=5, column=0, sticky="w")
        self.border_thr = tk.IntVar(value=30)
        fb = ttk.Frame(right); fb.grid(row=5, column=1, columnspan=3, sticky="w")
        ttk.Label(fb, text="cerceve esigi (0-255)").pack(side="left")
        ttk.Spinbox(fb, from_=5, to=120, increment=5, textvariable=self.border_thr, width=4).pack(side="left")
        ttk.Label(fb, text="  ok tuslari: secili koseyi 1 px oynatir (Shift: 10 px)").pack(side="left")
        ttk.Button(right, text="Ortala", command=self._crop_center).grid(row=2, column=2, sticky="w")
        self.mode = tk.StringVar(value="rect")
        fm = ttk.Frame(right); fm.grid(row=2, column=3, sticky="w")
        ttk.Radiobutton(fm, text="Dikdortgen", value="rect", variable=self.mode, command=self._mode_changed).pack(side="left")
        ttk.Radiobutton(fm, text="4 kose (yamuk/egik duzelt)", value="quad", variable=self.mode, command=self._mode_changed).pack(side="left")
        fo = ttk.Frame(right); fo.grid(row=3, column=0, columnspan=4, sticky="w", pady=(4, 0))
        self.btn_ok = ttk.Button(fo, text="Secimi ONAYLA (OK)", command=self._confirm); self.btn_ok.pack(side="left")
        self.btn_edit = ttk.Button(fo, text="Duzenle (geri al)", command=self._unconfirm, state="disabled"); self.btn_edit.pack(side="left", padx=6)
        self.state_lbl = ttk.Label(fo, text="", foreground="#b00"); self.state_lbl.pack(side="left")
        self.crop_lbl = ttk.Label(right, text="Resim yuklenmedi. Dosyayi buraya surukleyin veya Sec / Yukle.")
        self.crop_lbl.grid(row=4, column=0, columnspan=4, sticky="w")
        if HAS_DND:
            for wdg in (self.cv, e, self):
                try:
                    wdg.drop_target_register(DND_FILES)
                    wdg.dnd_bind("<<Drop>>", self._on_drop)
                except Exception:  # noqa: BLE001
                    pass
        self._say("Upscayl: " + (str(self.bin) if self.bin else "BULUNAMADI (yalniz Lanczos calisir). Kur: https://upscayl.org"))
        self._say("ImageMagick (16 bit/CMYK icin): " + (find_magick() or "BULUNAMADI"))
        self._say("Surukle-birak: " + ("acik" if HAS_DND else "kapali (pip install tkinterdnd2)"))
        self._pixels()

    # ------------------------------------------------------------ yardimci
    def _say(self, s: str) -> None:
        self.log.configure(state="normal"); self.log.insert("end", s + "\n"); self.log.see("end"); self.log.configure(state="disabled")

    def _drain(self) -> None:
        try:
            while True:
                m = self.q.get_nowait()
                if m == "\x00DONE":
                    self.pb.stop(); self._running = False
                    self.btn.configure(state="normal"); self.btn_stop.configure(state="disabled")
                    self.btn_restart.configure(state="normal")
                else:
                    self._say(m)
        except queue.Empty:
            pass
        self.after(200, self._drain)

    def _scale_factor(self) -> int:
        return int(self.scale_sel.get().rstrip("x"))

    def _target_cm(self) -> tuple[float, float]:
        """Plan icin cm: cm modunda alanlar; olcek modunda (kirpilmis) kaynak px x olcek / dpi."""
        if self.tmode.get() == "scale" and self.img:
            w, h = self._work_size()
            k = self._scale_factor()
            return w * k / self.dpi.get() * INCH_CM, h * k / self.dpi.get() * INCH_CM
        return self.wcm.get(), self.hcm.get()

    def _work_size(self) -> tuple[int, int]:
        """Kosuma girecek goruntunun boyutu (onayli ise islenmis, degilse secim)."""
        if not self.img:
            return (0, 0)
        if self.confirmed or self.mode.get() != "quad" and not self.crop:
            return self.img.size
        if self.mode.get() == "quad" and self.quad:
            return self._quad_out_size()
        if self.crop:
            return int(self.crop[2] - self.crop[0]), int(self.crop[3] - self.crop[1])
        return self.img.size

    def _pixels(self) -> None:
        try:
            if self.tmode.get() == "scale":
                if self.img:
                    w, h = self._work_size(); k = self._scale_factor()
                    self.pix.configure(text=f"= {w * k} x {h * k} px  ({w}x{h} x {k})")
                else:
                    self.pix.configure(text="(resim yukleyin)")
            else:
                w, h = target_pixels(self.wcm.get(), self.hcm.get(), self.dpi.get())
                self.pix.configure(text=f"= {w} x {h} px")
        except (tk.TclError, ValueError):
            self.pix.configure(text="")
        if self.crop and self.lock.get():
            self._apply_lock(); self._draw_crop()
        self._crop_info()

    def _preset(self, _=None) -> None:
        v = PRESETS.get(self.preset.get())
        if v:
            self.wcm.set(v[0]); self.hcm.set(v[1])

    def _pick_src(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("Resim", "*.jpg *.jpeg *.png *.tif *.tiff *.webp *.bmp"), ("Hepsi", "*")])
        if p:
            self.load_image(p)

    def _on_drop(self, ev) -> None:
        raw = ev.data.strip()
        # tkdnd: bosluklu yollar {..} icinde gelir; ilk dosyayi al
        if raw.startswith("{"):
            p = raw[1:raw.index("}")]
        else:
            p = raw.split()[0]
        self.load_image(p)

    def _pick_icc(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("ICC profili", "*.icc *.icm"), ("Hepsi", "*")])
        if p:
            self.icc.set(p)

    def _pick_out(self) -> None:
        p = filedialog.askdirectory()
        if p:
            self.outdir.set(p)

    # ------------------------------------------------------------ onizleme / kirpma
    def load_image(self, path: str) -> None:
        p = Path(path)
        if not p.is_file():
            messagebox.showerror("Hata", f"Dosya yok: {p}"); return
        try:
            im = Image.open(p); im.load()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Hata", f"Acilamadi: {e}"); return
        self.img = im.convert("RGB")
        self.img_orig = self.img
        self.confirmed = False
        self.src.set(str(p))
        self._show(self.img)
        self._crop_center()
        self._set_state()
        self._say(f"Yuklendi: {p.name} {self.img.size[0]}x{self.img.size[1]}px")

    def _show(self, im: Image.Image) -> None:
        """Onizlemeye verilen goruntuyu koy (kirpim cizimi ayri)."""
        self.img = im
        W, H = self.img.size
        self.scale = min(PREVIEW / W, PREVIEW / H)
        dw, dh = int(W * self.scale), int(H * self.scale)
        self.off = ((PREVIEW - dw) // 2, (PREVIEW - dh) // 2)
        self.disp = ImageTk.PhotoImage(self.img.resize((dw, dh), Image.LANCZOS))
        self.cv.delete("all")
        self.cv.create_image(self.off[0], self.off[1], anchor="nw", image=self.disp, tags="img")

    def _set_state(self) -> None:
        if self.confirmed:
            self.state_lbl.configure(text="  ONAYLI: bu goruntu islenecek", foreground="#080")
            self.btn_ok.configure(state="disabled"); self.btn_edit.configure(state="normal")
        else:
            self.state_lbl.configure(text="  duzenleniyor (OK'e basmadan Buyut = secim otomatik onaylanir)", foreground="#b00")
            self.btn_ok.configure(state="normal"); self.btn_edit.configure(state="disabled")
        self._pixels()

    def _confirm(self) -> None:
        """Secimi uygula: kirp/perspektif duzelt, sonucu onizlemede goster."""
        if not self.img_orig:
            return
        self.img = self.img_orig
        res = self._apply_selection()
        self.confirmed = True
        self._show(res)
        self.cv.delete("crop")
        self.crop = None; self.quad = None
        self._set_state()
        self.crop_lbl.configure(text=f"Onaylandi: {res.size[0]} x {res.size[1]} px. Degistirmek icin Duzenle.")
        self._say(f"Secim onaylandi: {res.size[0]}x{res.size[1]}px")

    def _unconfirm(self) -> None:
        if not self.img_orig:
            return
        self.confirmed = False
        self._show(self.img_orig)
        if self.mode.get() == "quad":
            W, H = self.img.size; self.quad = [[0, 0], [W, 0], [W, H], [0, H]]
            self._draw_crop()
        else:
            self._crop_center()
        self._set_state()

    def _apply_selection(self) -> Image.Image:
        """Mevcut secimi (dikdortgen veya 4 kose) orijinale uygula, PIL goruntu don."""
        im = self.img_orig
        assert im is not None
        W, H = im.size
        if self.mode.get() == "quad" and self.quad:
            w, h = self._quad_out_size()
            (ax, ay), (bx, by), (cx, cy), (dx, dy) = self.quad
            return im.transform((w, h), Image.QUAD, (ax, ay, dx, dy, cx, cy, bx, by), Image.BICUBIC)
        if self.crop:
            x0, y0, x1, y1 = (int(round(v)) for v in self.crop)
            if (x0, y0, x1, y1) != (0, 0, W, H):
                return im.crop((x0, y0, x1, y1))
        return im

    def _crop_all(self) -> None:
        if self.img:
            W, H = self.img.size; self.crop = [0, 0, W, H]
            if self.lock.get():
                self._apply_lock()
            self._draw_crop()

    def _crop_auto_border(self) -> None:
        """Siyah cerceveyi olcup kirpim kutusunu tam icerige oturt (oran kilidi gecici kapali)."""
        if not self.img_orig:
            return
        if self.confirmed:
            self._unconfirm()
        x0, y0, x1, y1 = find_border(self.img_orig, thresh=float(self.border_thr.get()))
        W, H = self.img_orig.size
        self.mode.set("rect")
        self.lock.set(False)
        self.crop = [x0, y0, x1, y1]
        self._draw_crop()
        self._say(f"Kenar bulundu: sol {x0} ust {y0} sag {W - x1} alt {H - y1} px cerceve -> kirpim {x1 - x0}x{y1 - y0}px "
                  f"(oran {(x1 - x0) / max(y1 - y0, 1):.3f}). Oran kilidi kapatildi; koseleri buyutecle kontrol et.")

    def _crop_center(self) -> None:
        """Baski oraninda, ortalanmis en buyuk dikdortgen."""
        if not self.img:
            return
        W, H = self.img.size
        ar = self._ratio()
        if W / H > ar:
            h = H; w = h * ar
        else:
            w = W; h = w / ar
        x0, y0 = (W - w) / 2, (H - h) / 2
        self.crop = [x0, y0, x0 + w, y0 + h]
        self._draw_crop()

    def _ratio(self) -> float:
        try:
            return max(1e-6, self.wcm.get() / self.hcm.get())
        except (tk.TclError, ZeroDivisionError):
            return 1.0

    def _crop_lock_changed(self) -> None:
        if self.crop and self.lock.get():
            self._apply_lock(); self._draw_crop()

    def _apply_lock(self, anchor: str = "c") -> None:
        """Mevcut kutuyu baski oranina getirir (merkezi koruyarak), goruntu icine sigdirir."""
        if not self.crop or not self.img:
            return
        W, H = self.img.size
        x0, y0, x1, y1 = self.crop
        ar = self._ratio()
        w, h = x1 - x0, y1 - y0
        if w / max(h, 1e-6) > ar:
            w = h * ar
        else:
            h = w / ar
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = min(w, W, H * ar), min(h, H, W / ar)
        x0, y0 = min(max(cx - w / 2, 0), W - w), min(max(cy - h / 2, 0), H - h)
        self.crop = [x0, y0, x0 + w, y0 + h]

    def _mode_changed(self) -> None:
        if not self.img:
            return
        if self.mode.get() == "quad":
            if self.crop:
                x0, y0, x1, y1 = self.crop
                self.quad = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            else:
                W, H = self.img.size; self.quad = [[0, 0], [W, 0], [W, H], [0, H]]
        else:
            if self.quad:
                xs = [p[0] for p in self.quad]; ys = [p[1] for p in self.quad]
                self.crop = [min(xs), min(ys), max(xs), max(ys)]
                if self.lock.get():
                    self._apply_lock()
        self._draw_crop()

    def _quad_out_size(self) -> tuple[int, int]:
        """Perspektif acilinca hedef dikdortgen: kenar uzunluklarinin ortalamasi; kilit acikken baski orani."""
        import math
        (ax, ay), (bx, by), (cx, cy), (dx, dy) = self.quad
        w = (math.hypot(bx - ax, by - ay) + math.hypot(cx - dx, cy - dy)) / 2
        h = (math.hypot(dx - ax, dy - ay) + math.hypot(cx - bx, cy - by)) / 2
        if self.lock.get():
            h = w / self._ratio()
        return max(8, int(round(w))), max(8, int(round(h)))

    def _s2c(self, x: float, y: float) -> tuple[float, float]:
        return self.off[0] + x * self.scale, self.off[1] + y * self.scale

    def _c2s(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.off[0]) / self.scale, (y - self.off[1]) / self.scale

    def _draw_crop(self) -> None:
        self.cv.delete("crop")
        if not self.img:
            return
        if self.mode.get() == "quad" and self.quad:
            pts = [self._s2c(x, y) for x, y in self.quad]
            flat = [v for p in pts for v in p]
            self.cv.create_polygon(*flat, outline="#ffd700", fill="", width=2, tags="crop")
            for hx, hy in pts:
                self.cv.create_rectangle(hx - HANDLE, hy - HANDLE, hx + HANDLE, hy + HANDLE, fill="#ffd700", outline="black", tags="crop")
            w, h = self._quad_out_size()
            try:
                eff = min(w / (self.wcm.get() / 2.54), h / (self.hcm.get() / 2.54))
                self.crop_lbl.configure(text=f"4 kose -> {w} x {h} px dik dikdortgene acilir (oran {w / h:.3f}); gercek cozunurluk ~{eff:.0f} dpi")
            except (tk.TclError, ZeroDivisionError):
                pass
            return
        if not self.crop:
            return
        W, H = self.img.size
        x0, y0 = self._s2c(self.crop[0], self.crop[1]); x1, y1 = self._s2c(self.crop[2], self.crop[3])
        X0, Y0 = self._s2c(0, 0); X1, Y1 = self._s2c(W, H)
        for box in ((X0, Y0, X1, y0), (X0, y1, X1, Y1), (X0, y0, x0, y1), (x1, y0, X1, y1)):
            self.cv.create_rectangle(*box, fill="black", stipple="gray50", outline="", tags="crop")
        self.cv.create_rectangle(x0, y0, x1, y1, outline="#ffd700", width=2, tags="crop")
        for hx, hy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
            self.cv.create_rectangle(hx - HANDLE, hy - HANDLE, hx + HANDLE, hy + HANDLE, fill="#ffd700", outline="black", tags="crop")
        self._crop_info()

    def _crop_info(self) -> None:
        if not self.crop or not self.img:
            return
        w, h = self.crop[2] - self.crop[0], self.crop[3] - self.crop[1]
        try:
            eff = min(w / (self.wcm.get() / 2.54), h / (self.hcm.get() / 2.54))
            self.crop_lbl.configure(text=f"Kirpim: {w:.0f} x {h:.0f} px  (oran {w / max(h, 1):.3f}); bu boyutta gercek cozunurluk ~{eff:.0f} dpi")
        except (tk.TclError, ZeroDivisionError):
            pass

    def _hit(self, cx: float, cy: float):
        if self.mode.get() == "quad" and self.quad:
            for i, (qx, qy) in enumerate(self.quad):
                hx, hy = self._s2c(qx, qy)
                if abs(cx - hx) <= HANDLE + 3 and abs(cy - hy) <= HANDLE + 3:
                    return f"q{i}"
            xs = [p[0] for p in self.quad]; ys = [p[1] for p in self.quad]
            sx, sy = self._c2s(cx, cy)
            return "qmove" if (min(xs) <= sx <= max(xs) and min(ys) <= sy <= max(ys)) else None
        if not self.crop:
            return None
        x0, y0 = self._s2c(self.crop[0], self.crop[1]); x1, y1 = self._s2c(self.crop[2], self.crop[3])
        for name, (hx, hy) in {"nw": (x0, y0), "ne": (x1, y0), "sw": (x0, y1), "se": (x1, y1)}.items():
            if abs(cx - hx) <= HANDLE + 3 and abs(cy - hy) <= HANDLE + 3:
                return name
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            return "move"
        return None

    def _draw_loupe(self, cx: float, cy: float) -> None:
        """Imlecin altindaki kaynak bolgesini LOUPE_ZOOM kat buyutup canvas kosesine ciz; artı isareti = tam nokta."""
        self.cv.delete("loupe")
        if not self.img:
            return
        sx, sy = self._c2s(cx, cy)
        W, H = self.img.size
        n = LOUPE / LOUPE_ZOOM              # kaynak px cinsinden pencere
        bx0, by0 = sx - n / 2, sy - n / 2
        box = (int(round(bx0)), int(round(by0)), int(round(bx0 + n)), int(round(by0 + n)))
        tile = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (255, 0, 255))   # dis alan macenta
        ix0, iy0 = max(box[0], 0), max(box[1], 0); ix1, iy1 = min(box[2], W), min(box[3], H)
        if ix1 > ix0 and iy1 > iy0:
            tile.paste(self.img.crop((ix0, iy0, ix1, iy1)), (ix0 - box[0], iy0 - box[1]))
        tile = tile.resize((LOUPE, LOUPE), Image.NEAREST)
        self._loupe_img = ImageTk.PhotoImage(tile)
        # imlec sag-altta ise sol-uste, degilse sag-uste koy
        lx = 6 if cx > PREVIEW / 2 else PREVIEW - LOUPE - 6
        ly = 6
        self.cv.create_image(lx, ly, anchor="nw", image=self._loupe_img, tags="loupe")
        self.cv.create_rectangle(lx, ly, lx + LOUPE, ly + LOUPE, outline="#ffd700", width=2, tags="loupe")
        m = lx + LOUPE / 2; k = ly + LOUPE / 2
        self.cv.create_line(m - 14, k, m + 14, k, fill="#ff3030", width=1, tags="loupe")
        self.cv.create_line(m, k - 14, m, k + 14, fill="#ff3030", width=1, tags="loupe")
        # kirpim kutusunun kenarlarini buyutec icinde de goster
        if self.crop:
            for v, vertical in ((self.crop[0], True), (self.crop[2], True), (self.crop[1], False), (self.crop[3], False)):
                if vertical:
                    px = lx + (v - box[0]) * LOUPE_ZOOM
                    if lx <= px <= lx + LOUPE:
                        self.cv.create_line(px, ly, px, ly + LOUPE, fill="#ffd700", width=1, tags="loupe")
                else:
                    py = ly + (v - box[1]) * LOUPE_ZOOM
                    if ly <= py <= ly + LOUPE:
                        self.cv.create_line(lx, py, lx + LOUPE, py, fill="#ffd700", width=1, tags="loupe")
        self.cv.create_text(lx + 4, ly + LOUPE - 4, anchor="sw", fill="#ffd700", tags="loupe",
                            text=f"x{LOUPE_ZOOM}  ({sx:.0f}, {sy:.0f})")

    def _cv_motion(self, ev) -> None:
        if self.img and not self.confirmed:
            self._draw_loupe(ev.x, ev.y)

    def _nudge(self, ev) -> None:
        """Ok tuslari: secili koseyi 1 px (Shift ile 10 px) oynat."""
        if not self.img or not self.active_corner or self.confirmed:
            return
        step = 10 if (ev.state & 0x1) else 1
        dx = {"Left": -step, "Right": step}.get(ev.keysym, 0)
        dy = {"Up": -step, "Down": step}.get(ev.keysym, 0)
        W, H = self.img.size
        c = self.active_corner
        if self.mode.get() == "quad" and self.quad and c.startswith("q"):
            i = int(c[1]); self.quad[i][0] = min(max(self.quad[i][0] + dx, 0), W); self.quad[i][1] = min(max(self.quad[i][1] + dy, 0), H)
        elif self.crop and c in ("nw", "ne", "sw", "se"):
            x0, y0, x1, y1 = self.crop
            if "w" in c: x0 = min(max(x0 + dx, 0), x1 - 1)
            else: x1 = max(min(x1 + dx, W), x0 + 1)
            if "n" in c: y0 = min(max(y0 + dy, 0), y1 - 1)
            else: y1 = max(min(y1 + dy, H), y0 + 1)
            self.crop = [x0, y0, x1, y1]
        else:
            return
        self._draw_crop()
        # buyuteci o koseye tasi
        if self.mode.get() == "quad" and self.quad and c.startswith("q"):
            px, py = self.quad[int(c[1])]
        else:
            px = self.crop[0] if "w" in c else self.crop[2]; py = self.crop[1] if "n" in c else self.crop[3]
        cx, cy = self._s2c(px, py); self._draw_loupe(cx, cy)

    def _cv_press(self, ev) -> None:
        if not self.img:
            return
        self.cv.focus_set(); self.focus_set()
        mode = self._hit(ev.x, ev.y)
        if mode and mode not in ("move", "qmove"):
            self.active_corner = mode
            self.crop_lbl.configure(text=f"Secili kose: {mode}  (ok tuslari 1 px, Shift 10 px)")
        sx, sy = self._c2s(ev.x, ev.y)
        if self.mode.get() == "quad":
            if mode is None or not self.quad:
                return
            self._drag = (mode, sx, sy, [list(p) for p in self.quad]); return
        if mode is None:
            self.crop = [sx, sy, sx, sy]; mode = "new"
        self._drag = (mode, sx, sy, list(self.crop))

    def _cv_drag(self, ev) -> None:
        if not self._drag or not self.img:
            return
        mode, sx0, sy0, c0 = self._drag
        W, H = self.img.size
        sx, sy = self._c2s(ev.x, ev.y)
        sx, sy = min(max(sx, 0), W), min(max(sy, 0), H)
        dx, dy = sx - sx0, sy - sy0
        if self.mode.get() == "quad":
            if mode == "qmove":
                self.quad = [[min(max(x + dx, 0), W), min(max(y + dy, 0), H)] for x, y in c0]
            elif mode and mode.startswith("q"):
                i = int(mode[1]); self.quad = [list(p) for p in c0]; self.quad[i] = [sx, sy]
            self._draw_crop(); self._draw_loupe(ev.x, ev.y); return
        x0, y0, x1, y1 = c0
        if mode == "move":
            w, h = x1 - x0, y1 - y0
            nx0 = min(max(x0 + dx, 0), W - w); ny0 = min(max(y0 + dy, 0), H - h)
            self.crop = [nx0, ny0, nx0 + w, ny0 + h]
        else:
            if mode == "new":
                x0, y0, x1, y1 = min(sx0, sx), min(sy0, sy), max(sx0, sx), max(sy0, sy)
            elif mode == "se": x1, y1 = sx, sy
            elif mode == "nw": x0, y0 = sx, sy
            elif mode == "ne": x1, y0 = sx, sy
            elif mode == "sw": x0, y1 = sx, sy
            x0, x1 = min(x0, x1), max(x0, x1); y0, y1 = min(y0, y1), max(y0, y1)
            if self.lock.get():
                ar = self._ratio()
                w, h = x1 - x0, y1 - y0
                if w / max(h, 1e-6) > ar: w = h * ar
                else: h = w / ar
                if mode in ("nw", "sw"): x0 = x1 - w
                else: x1 = x0 + w
                if mode in ("nw", "ne"): y0 = y1 - h
                else: y1 = y0 + h
                x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W), min(y1, H)
            self.crop = [x0, y0, x1, y1]
            if self.lock.get():
                self._apply_lock()
        self._draw_crop()
        self._draw_loupe(ev.x, ev.y)

    def _cv_release(self, ev) -> None:
        if self.mode.get() == "quad":
            self._drag = None; return
        if self.crop and (self.crop[2] - self.crop[0] < 8 or self.crop[3] - self.crop[1] < 8):
            self._crop_center()
        self._drag = None

    def _cropped_source(self) -> Path:
        """Kosuma girecek dosya: onayli ise islenmis goruntu; degilse secim otomatik onaylanir."""
        src = Path(self.src.get())
        if not self.img_orig:
            return src
        if not self.confirmed:
            self._confirm()
        assert self.img is not None
        if self.img is self.img_orig:
            return src
        out = Path(tempfile.gettempdir()) / f"imgup_sel_{src.stem}.png"
        self.img.save(out, "PNG")
        self._say(f"Islenecek: {self.img.size[0]}x{self.img.size[1]}px -> {out}")
        return out

    # ------------------------------------------------------------ kosum
    def _run(self) -> None:
        if not self.img and Path(self.src.get()).is_file():
            self.load_image(self.src.get())
        if not self.img:
            messagebox.showerror("Hata", "Once bir resim yukleyin"); return
        orig = Path(self.src.get())
        src = self._cropped_source()
        wcm, hcm = self._target_cm()
        try:
            plan = make_plan(src, wcm, hcm, self.dpi.get(),
                             self.model.get(), self.passes.get(), self.fit.get(),
                             self.bits.get(), self.color.get(),
                             Path(self.icc.get()) if self.icc.get() else None)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Hata", str(e)); return
        if self.dedge.get():
            plan.darken_edges = RECOMMENDED["darken_edges"]; plan.edge_reflect = RECOMMENDED["edge_reflect"]
        if self.pr_on.get() or self.pr_grain.get() > 0 or self.ftone.get() != 0.0 or self.fsd.get() > 0:
            plan.photoreal = {"face_exposure": self.ftone.get() or None, "face_gamma": None,
                              "face_sd": self.fsd.get() or None,
                              "sd_denoise": self.pr_denoise.get() if self.pr_on.get() else None,
                              "sd_tile": RECOMMENDED["sd_tile"], "sd_steps": RECOMMENDED["sd_steps"],
                              "face_fidelity": None, "face_blend": 0.5,
                              "grain": self.pr_grain.get() or None, "mem_limit_gb": self.mem.get()}
        ext = {"tiff": "tif", "jpg": "jpg", "png": "png"}[self.fmt.get()]
        tag = f"_{self.color.get()}{self.bits.get()}" if (self.color.get() != "rgb" or self.bits.get() != 8) else ""
        if self.pr_on.get():
            tag += "_photoreal"
        size_tag = f"{self._scale_factor()}x" if self.tmode.get() == "scale" else f"{wcm:g}x{hcm:g}cm"
        out = Path(self.outdir.get()) / f"{orig.stem}_{size_tag}_{self.dpi.get()}dpi{tag}.{ext}"
        self._running = True
        self.btn.configure(state="disabled"); self.btn_stop.configure(state="normal"); self.btn_restart.configure(state="disabled")
        self.pb.start(10)

        def work() -> None:
            try:
                upscale(src, out, plan, self.bin, fmt=self.fmt.get(), log=self.q.put)
                self.q.put("BITTI. Sonucu 100% yakinlikta kontrol et.")
            except Cancelled:
                self.q.put("DURDURULDU (alt surecler oldu, gecici dosyalar silindi).")
            except Exception as e:  # noqa: BLE001
                self.q.put(f"HATA: {e}")
            finally:
                self.q.put("\x00DONE")

        threading.Thread(target=work, daemon=True).start()


    def _stop(self) -> None:
        if self._running:
            self._say("Durduruluyor ...")
            cancel()

    def _restart(self) -> None:
        if self._running:
            cancel()
            self.after(800, self._restart); return
        self._run()


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
