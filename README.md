# ImageUpscaler

Baski icin goruntu buyutme araci. Hedef **baski boyutu (cm) + dpi** verilir; arac gerekli piksel
sayisini hesaplar, Real-ESRGAN ile 4x AI gecisi yapar, kalan orani Lanczos ile alir ve dpi bilgisi
gomulu TIFF / JPEG / PNG yazar. Tamamen yerel calisir; goruntu buluta gitmez.

Ozellikler:
- GUI (surukle-birak, kirpma, 4 kose perspektif duzeltme, siyah cerceve bulma, buyutec) ve CLI.
- Hedef: cm + dpi ya da 2x / 4x / 8x / 16x olcek.
- 8 / 16 bit, sRGB veya CMYK (ICC profili gomulu) TIFF.
- Istege bagli fotogercekci asama: Stable Diffusion 1.5 + ControlNet Tile ile cilt / sac / kumas dokusu, film greni.
- Tarama kenari beyazliklarini icteki tona uyduran kenar duzeltme, yuz tonu ayari.
- Bellek bekcisi: belirlenen RAM limiti asilirsa is kendini durdurur, makine swap'e dusmez.
- Durdur / yeniden baslat; takilan alt surecler otomatik yeniden denenir.

---

## Kurulum

Iki katman vardir. **Temel katman** (AI 4x buyutme + baski ciktisi) birkac dakikada kurulur.
**Fotogercekci katman** (SD doku, yuz tonu) istege baglidir; torch ve ~5 GB model ister.

### Ortak: Python ve Upscayl

1. **Python 3.10 veya ustu** (3.12 onerilir).
2. **Upscayl** (Real-ESRGAN ncnn motoru ve modelleri onunla gelir): https://upscayl.org
   Arac Upscayl'in `upscayl-bin` ikilisini ve model klasorunu kendi bulur. Bulamazsa
   `UPSCAYL_BIN` ve `UPSCAYL_MODELS` ortam degiskenleriyle yol verilir. Upscayl yoksa arac
   yalniz Lanczos ile calisir (AI gecisi atlanir).
3. Depoyu indir ve bagimliliklari kur:
   ```bash
   git clone https://github.com/srhtbynkln/ImageUpscaler
   cd ImageUpscaler
   pip install -r requirements.txt
   ```
   `requirements.txt` yalniz Pillow ve (surukle-birak icin) tkinterdnd2 icerir.
4. 16 bit ve CMYK cikti icin **ImageMagick 7** (`magick` komutu PATH'te ya da `MAGICK_BIN`).

### Windows

```bat
winget install Python.Python.3.12
winget install Upscayl.Upscayl
winget install ImageMagick.ImageMagick        :: yalniz 16 bit / CMYK icin
git clone https://github.com/srhtbynkln/ImageUpscaler
cd ImageUpscaler
py -3 -m pip install -r requirements.txt
```
- Baslatma: `start.bat` cift tikla ya da `py -3 -m imageupscaler.gui`.
- Upscayl varsayilan yolu `%LOCALAPPDATA%\Programs\Upscayl\resources\bin\upscayl-bin.exe`
  ya da `C:\Program Files\Upscayl\...`. Farkliysa:
  ```bat
  setx UPSCAYL_BIN "C:\yol\upscayl-bin.exe"
  setx UPSCAYL_MODELS "C:\yol\models"
  ```
- CMYK profilleri `C:\Windows\System32\spool\drivers\color` ve Adobe klasorlerinden bulunur.
- Fotogercekci katman icin NVIDIA GPU ile CUDA'li torch kurulmasi onerilir (asagida).

### macOS

```bash
brew install python@3.12
brew install --cask upscayl
brew install imagemagick                       # yalniz 16 bit / CMYK icin
git clone https://github.com/srhtbynkln/ImageUpscaler
cd ImageUpscaler
python3 -m pip install -r requirements.txt
```
- Baslatma: `start.command` cift tikla ya da `python3 -m imageupscaler.gui`.
- Upscayl `/Applications/Upscayl.app` icinden bulunur.
- Apple Silicon'da fotogercekci katman Metal (MPS) uzerinde calisir; 16 GB RAM ile 512 px karo onerilir.

### Linux

```bash
sudo apt install python3 python3-pip python3-tk imagemagick   # Debian/Ubuntu; tk GUI icin sart
git clone https://github.com/srhtbynkln/ImageUpscaler
cd ImageUpscaler
python3 -m pip install -r requirements.txt
```
- Upscayl: https://upscayl.org adresinden `.deb` / `.rpm` / AppImage. `.deb` kurulumu
  `/opt/Upscayl/resources/bin/upscayl-bin` yolunu kullanir ve otomatik bulunur. AppImage kullaniyorsan
  paketi ac (`./Upscayl.AppImage --appimage-extract`) ve `UPSCAYL_BIN` / `UPSCAYL_MODELS` ile yol ver.
  Alternatif: `realesrgan-ncnn-vulkan` PATH'teyse o da kullanilir.
- Vulkan surucusu gerekir (`vulkaninfo` calismali).
- Baslatma: `./start.sh` ya da `python3 -m imageupscaler.gui`.

### Fotogercekci katman (istege bagli, her platform)

Ayri bir sanal ortam (`.venv`, depo kokunde) kurulur; arac bu ortami kendi bulur ve fotogercekci
asamayi ayri surecte kosar.

```bash
# uv ile (onerilen): https://docs.astral.sh/uv
uv venv --python 3.12 .venv
uv pip install -r requirements-photoreal.txt
# ya da klasik:
python3 -m venv .venv && .venv/bin/pip install -r requirements-photoreal.txt   # Windows: .venv\Scripts\pip
```
- **NVIDIA GPU (Windows / Linux):** torch'u CUDA ile kur, ornek
  `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` (surum icin pytorch.org).
- **Apple Silicon:** varsayilan torch MPS destekler, ek adim yok.
- **Yalniz CPU:** calisir ama 2400 px kare icin saatler surer.
- SD modelleri ilk kosumda Hugging Face'ten iner (~5 GB): `SG161222/Realistic_Vision_V5.1_noVAE`,
  `stabilityai/sd-vae-ft-mse`, `lllyasviel/control_v11f1e_sd15_tile`. Baska model icin
  `UPSCALER_SD_BASE`, `UPSCALER_SD_VAE`, `UPSCALER_SD_TILE`.
- **Yuz bulma (yuz tonu, yuz SD) ve CodeFormer icin** CodeFormer deposu ve agirliklari:
  ```bash
  git clone --depth 1 https://github.com/sczhou/CodeFormer third_party/CodeFormer
  mkdir -p weights && cd weights
  curl -fLO https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
  curl -fLO https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth
  curl -fLO https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth
  ```
  Agirliklar ilk kullanimda CodeFormer'in bekledigi klasorlere baglanir (Windows'ta kopyalanir).
  `codeformer.pth` 359 MB olmali; kucukse indirme kirik demektir.

---

## Kullanim

### GUI
```bash
python -m imageupscaler.gui        # Windows: py -3 -m imageupscaler.gui
```
1. Goruntuyu pencereye surukle ya da **Sec / Yukle**.
2. Gerekirse kirp: fareyle kutu ciz (kose tutamaci boyut, ic surukleme tasima). **Oran kilidi**
   kutuyu baski oraninda tutar, **Ortala** baski oraninda en buyuk kutu, **Tumu** tam resim.
   Siyah cerceveli taramada **Kenar bul** cerceveyi olcup kutuyu iceri oturtur; **buyutec** imlecin
   altini x6 gosterir; kose secili iken ok tuslari 1 px (Shift ile 10 px) oynatir.
   **4 kose** modunda her kose ayri suruklenir; egik / yamuk tarama dik dikdortgene acilir.
3. **Secimi ONAYLA**: secim uygulanir, duzeltilmis goruntu onizlemede gorunur (**Duzenle** geri alir).
4. Hedef: **cm + dpi** (ornek 100 x 100 cm, 312 dpi) ya da **olcek** 2x / 4x / 8x / 16x.
5. Model (fotograf icin `high-fidelity-4x`), AI gecisi sayisi, format (TIFF / JPEG / PNG), 8 / 16 bit, sRGB / CMYK.
6. **Buyut**. Alt pencerede ilerleme ve her adimin sadakat olcumu (dB) yazilir. **Durdur** alt surecleri
   oldurur, **Yeniden baslat** ayni ayarla tekrar kosar.

Onerilen varsayilanlar GUI'de yuklu gelir. Kisa rehber:

| Goruntu | Ayar |
|---|---|
| Portre, kaynak kucuk (< 1000 px), fotografik doku isteniyor | SD doku 0.25 - 0.30, gren 0.02 - 0.03 |
| Buyuk tarama (> 3000 px), duz acik fon | SD doku **0** (kapali), gren 0.02 |
| Tarama kenarlarinda beyaz seritler | kenar duzelt acik |
| Yuz bir tik parlak | yuz ton -0.05 ... -0.10 |

### CLI
```bash
python -m imageupscaler.cli girdi.jpg --cm 100x100 --dpi 312 --recommended -o cikti.tif
python -m imageupscaler.cli girdi.jpg --cm 74.3x100 --dpi 312 --fit crop --grain 0.02 -o cikti.tif
python -m imageupscaler.cli girdi.jpg --cm 30x20 --dpi 300 --bits 16 --color cmyk --cmyk-icc ISOcoated_v2.icc -o cikti.tif
python -m imageupscaler.cli --list-models
python -m imageupscaler.cli --list-cmyk
```
Onemli bayraklar: `--model`, `--passes` (AI gecisi sayisi), `--fit fit|crop|stretch`, `--bits 8|16`,
`--color rgb|cmyk`, `--photoreal --sd-denoise 0.30 --sd-tile 512 --sd-steps 20`, `--grain 0.03`,
`--face-exposure -0.10`, `--face-sd 0.40`, `--darken-edges UST:ALT:SOL:SAG:GUC`, `--mem-limit-gb 12`.

---

## Nasil calisir

### AI gecisi
Real-ESRGAN tabanli model goruntuyu tek seferde 4 kat buyutur; her piksel icin cevresine bakarak
"burada ne olmaliydi" tahmini yapar, JPEG bloklarini temizler, kenarlari toparlar. Bu bir **tahmindir**:
kaynakta olmayan doku uretebilir. Iki gecis ust uste (16x) uydurmayi katlar (600 px portrede ikinci gecis
alinda catlak benzeri sahte doku uretti). Varsayilan **1 gecis**, kalan oran yorumsuz Lanczos ile alinir.

### Neden 4x
Upscayl ile gelen modellerin hepsi sabit 4x agdir. 2x istenirse 4x kosulup kucultulur (Upscayl da
boyle yapar); 8x / 16x gecis zinciridir. Hedef piksel cm + dpi ile serbesttir. Baska model klasoru
`UPSCAYL_MODELS` ile verilirse oradaki `.param/.bin` ciftleri listeye girer.

### Model secimi
| Model | Ne zaman |
|---|---|
| high-fidelity-4x | Fotograf, portre; en az uydurma. **Varsayilan** |
| remacri-4x | Manzara / doku; keskin ama ahsap, kumas gibi yerlerde damar uydurur |
| ultrasharp-4x | Grafik / yazi; fotografta asiri keskin |
| upscayl-standard-4x | Genel; yuzu plastiklestirir |
| digital-art-4x | Cizim / illustrasyon |

### Buyuk kaynaklar
Kaynak hedef / 4'ten buyukse once Lanczos ile hedef / 4'e kucultulur, sonra AI 4x kosar (tam boy
AI gecisi 18k x 24k gibi boyutlarda bellegi asar). `--fit crop` ise once hedef oranina kirpilir.
SD doku 1500 px ve ustu kaynakta AI'dan ONCE kosar, 3200 px ustunde atlanir.

### 16 bit ve CMYK
- `--bits 16`: son Lanczos adimi ImageMagick Q16 icinde 16 bit hesaplanir. 8 bit kaynaktan 16 bit
  cikti yeni bilgi eklemez; matbaa hatti istiyorsa kullanilir, tonlama bantlasmaz.
- `--color cmyk`: sRGB -> CMYK donusumu ICC profiliyle yapilir ve profil TIFF'e gomulur. Profil
  `--cmyk-icc`, GUI'den ya da `UPSCALER_CMYK_ICC`. Verilmezse makinede bulunan ilk aday; matbaanin
  profili (cogu zaman **ISO Coated v2 / FOGRA39**, eci.org) dogrusudur.

### Fotogercekci asama
Restoratif zincir kaynaga sadik ama duz / boyanmis gorunur. `--photoreal` (GUI: **SD doku**) AI 4x
ciktisinin ustune SD1.5 (Realistic Vision 5.1) + ControlNet Tile ile karo bazli img2img kosar: yapi
ControlNet ile sabitlenir, yalniz doku uretilir. Ardindan film greni.

Olcum (600 px sepya portre, kaynaga PSNR):
| Yol | Sadakat | Gorunum |
|---|---|---|
| Real-ESRGAN high-fidelity | 33,4 dB | temiz ama boyanmis |
| + SD doku 0.30 + gren 0.03 | 27,4 dB | fotografik, kirisiklar korunur |
| Bulut uretken model (kiyas) | 25,0 dB | fotografik, ayrintilar degismis |

- `--sd-denoise` 0.2 - 0.3 yalniz doku; 0.4 ve ustu bicim de degisir.
- `--sd-tile 512` 16 GB icin; 768 daha az ek ama daha cok bellek.
- Sure: Apple M2 16 GB'da 2392 px kare = 36 karo, yaklasik 9 dakika, tepe RSS 4 - 5 GB.
- Her karo (x, y) konumuna bagli sabit tohumla uretilir: ayni girdi + ayni ayar = ayni cikti.
- SD ciktisinin dusuk frekansi (ton, golge) girdiden geri alinir (`lf_restore`); boylece duz acik
  fonlarda karo izgarasi leke / dalga olarak gorunmez, yalniz doku kalir.
- `--face-fidelity` (CodeFormer) varsayilan kapali: yuzu genclestirip kimligi kaydirir. Yalniz cok
  bozuk yuzlerde `--face-blend 0.3 - 0.5` ile harmanlayarak dene.
- Duz, tek renk fonlu buyuk taramalarda SD dokuya gerek yoktur; kapali tut.

### Yuz
- `--face-exposure -0.10` (GUI: **yuz ton**): yuz otomatik bulunur (RetinaFace, yoksa Haar), eliptik
  yumusak maskeyle yuz %10 koyulasir; + deger aydinlatir; `--face-gamma` orta tonlar.
- `--face-sd 0.40` (GUI: **yuz SD**): yuz bolgesine portre-isigi istemiyle ayri, daha guclu SD gecisi;
  goz alti / yanak golgesi uretir ama kas ve kirisiklari kaynaktan uzaklastirir. Varsayilan kapali.
- Global ton egrisi (`--highlight`, `--contrast`) yuzu grilestirdigi icin onerilmez.

### Kenar duzeltme
`--darken-edges UST:ALT:SOL:SAG:GUC` (GUI: **kenar duzelt**): tarama kenarindaki beyazliklari o
bolgenin ic tonuna uydurur. En dis serit icteki dokunun kaydirilmis kopyasiyla doldurulur (capraz
gecisli; gren surekli kalir), bantta parlak ton rampayla icteki referansa cekilir, parlak yonlu gren
kirpilir. Ornek `0.012:0.004:0.010:0.008:1`. Yansitma seridi genisligi kodda `reflect=`.

### Bellek bekcisi
Fotogercekci asama ayri surecte kosar; `--mem-limit-gb` (varsayilan 12, GUI: **bellek limiti**)
asilirsa surec kendini `exit 3` ile keser, ana program "bekci durdurdu" der. Apple Silicon'da Metal
ayirmalarina da tavan konur. Ortam degiskeni: `UPSCALER_MEM_LIMIT_GB`.

### Takilma bekcisi
`upscayl-bin` GPU baslatirken nadiren sessizce asili kalabilir. Alt surec 120 saniye boyunca hicbir
cikti yazmazsa oldurulur ve iki kez yeniden denenir; sonra hata verilir.

---

## Sorun giderme

| Belirti | Cozum |
|---|---|
| "Upscayl bulunamadi", yalniz Lanczos | Upscayl kur; ya da `UPSCAYL_BIN` + `UPSCAYL_MODELS` ver |
| Linux'ta `No module named tkinter` | `sudo apt install python3-tk` |
| Surukle-birak calismiyor | `pip install tkinterdnd2`; yoksa **Sec / Yukle** kullan |
| "Fotogercekci asama icin repo/.venv yok" | Fotogercekci katman kurulumunu yap (yukarida) |
| Alt surec `exit 3` ile bitti | Bellek limiti asildi; `--sd-tile 512`, `--mem-limit-gb` artir ya da kaynak kucult |
| Alt surec `-9` / sessiz oldu | RAM yetmedi; kaynak hedef / 4'e kucultulmus mu logda kontrol et |
| Windows'ta `magick` bulunamadi | ImageMagick kurulumunda "Add to PATH" isaretle ya da `MAGICK_BIN` |
| CMYK profili yok | `--list-cmyk`; matbaanin `.icc` dosyasini `--cmyk-icc` ile ver |
| Ilk SD kosumu cok uzun | Model indiriliyor (~5 GB); ikinci kosumda onbellekten gelir |

---

## Baski notlari
- Cikti TIFF LZW, 8 bit sRGB, ICC profili kaynaktan tasinir, dpi TIFF etiketinde.
- Kaynagin gercek cozunurlugu (px / baski eni) uyari olarak yazilir; 600 px kaynak 1 m'de ~15 dpi'dir.
  Buyutme kaybolan detayi geri getirmez, pikselleri gorunmez kilar. Sonucu daima **%100 yakinlikta** kontrol et.
- 1 m x 1 m gibi buyuk baskilarda izleme mesafesi 1,5 - 2 m'dir; 150 dpi bile yeterlidir, 312 dpi
  cogu matbaanin sablon istegidir.
- Uretken (bulut) bir modelin ciktisi da bu aracla hedef dpi'ye getirilebilir; boyle bir kaynakta AI
  gecisi JPEG kalintilarini catlak dokuya cevirebilir, `--passes 0` (yalniz Lanczos) ya da dusuk SD doku dene.

## Lisans
MIT. Upscayl, Real-ESRGAN, CodeFormer, Stable Diffusion ve ControlNet kendi lisanslarina tabidir.
