import os, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# # =========== CONFIG ===========
# # INPUT_DIR   = "/home/maria/Disco2/clasificadas_nuevas/Nodulo"
# # EXTS        = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff")

# USE_CLAHE   = True
# CLAHE_CLIP  = 1.5
# CLAHE_TILE  = 8

# K_GRAD      = 11
# SIGMA_DETAIL= 1.0
# LAMBDA      = 0.6
# BETA        = 0.7
# RIM_PX      = 20
# MAX_DIM     = 1600
# # =============================

# def list_images(folder):
#     paths = []
#     for ext in EXTS: paths.extend(glob.glob(os.path.join(folder, ext)))
#     return sorted(list(dict.fromkeys(paths)))

# def maybe_resize(img, max_dim):
#     if max_dim is None: return img
#     H,W = img.shape[:2]; m = max(H,W)
#     if m <= max_dim: return img
#     s = max_dim/float(m)
#     return cv2.resize(img, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA)

# def to_gray01(img):
#     # if img.ndim==3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = img.astype(np.float32)
#     lo,hi = np.percentile(img,(0.5,99.5))
#     return np.clip((img-lo)/max(1e-6,hi-lo),0,1)

# def clahe01(img01, clip=1.5, tile=8):
#     u8 = (np.clip(img01,0,1)*255).astype(np.uint8)
#     out = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile,tile)).apply(u8)
#     return out.astype(np.float32)/255.0

# #separa tejido mamario del fondo para no potenciar el contorno etc
# def tissue_mask(img01):
#     blur = cv2.GaussianBlur(img01,(0,0),2.0) #suaviza variaciones finas
#     u8 = (blur*255).astype(np.uint8)
#     thr,_ = cv2.threshold(u8,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  #umbral automático que separa fondo (oscuro) y tejido (claro)
#     m = (u8>thr).astype(np.uint8) #m=1 tejido, m=0 fondo
#     k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
#     m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k) #cierre: rellena pequeños agujeros en la máscara
#     m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k) #apertura: elimina puntitos aislados brillantes fuera de la mama.
#     return m

# #desactiva el borde externo de la mama (gradientes altos y no contienen información de nodulos)
# def inner_mask_by_distance(mask, rim_px):
#     if rim_px <= 0: return mask
#     dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3) #distancia euclídea hasta el fondo más cercano
#     inner = (dist > rim_px).astype(np.uint8)
#     return (inner & mask).astype(np.uint8)

# #indica cuanta energia de borde hay alrededor de cada pixel (k=11 detalle fino)
# def grad_mean(img01, k):
#     gx = cv2.Sobel(img01, cv2.CV_32F, 1, 0, ksize=3)
#     gy = cv2.Sobel(img01, cv2.CV_32F, 0, 1, ksize=3) #gradiente con Sobel (cambios de intensidad)
#     grad = cv2.magnitude(gx, gy)
#     gmean = cv2.boxFilter(grad, -1, (k,k), normalize=True) #se promedia con una ventana kxk 
#     p = np.percentile(gmean, 99.0) #se recorta por percentil y se normaliza [0,1]
#     gmean = np.clip(gmean, 0, max(p,1e-6))
#     gmean /= (gmean.max()+1e-8)
#     return gmean

# #elimina las bajas frecuencias y deja las altas frecuencias (bordes finos, texturas etc.)
# def detail_layer(img01, sigma=1.0):
#     blur = cv2.GaussianBlur(img01,(0,0),sigma)
#     D = img01 - blur
#     p = np.percentile(np.abs(D), 95.0) + 1e-6
#     return D / p

# #realzar bordes de nodulos
# def enhance(img01, gmean, mask_inner, lam=0.6, beta=0.7, sigma_detail=1.0):
#     Dn = detail_layer(img01, sigma_detail) #donde hay detalle
#     G  = gmean * mask_inner.astype(np.float32) #donde hay borde coherente y dentro del tejido
#     enh = img01 + lam * (G * Dn) #en bordes se añade mas detalle (G alto) y en interiores homogeneos no cambia (G bajo)
#     enh = np.clip(enh, 0, 1)
#     out = (1.0-beta)*img01 + beta*enh #realce en el que añadimos solo micro-contraste al rededor de bordes
#     return out

# def show_triplet(img, g_masked, enhanced, title=None):
#     plt.figure(figsize=(15,5))
#     for i,(im,tt) in enumerate([
#         (img, "Original (norm+CLAHE)" if USE_CLAHE else "Original (norm)"),
#         (g_masked, f"Grad-mean k={K_GRAD} (enmasc.)"),
#         (enhanced, f"Enhanced (λ={LAMBDA}, β={BETA}, σd={SIGMA_DETAIL})"),
#     ], 1):
#         ax = plt.subplot(1,3,i)
#         ax.imshow(im, cmap="gray", vmin=0, vmax=1)
#         ax.set_title(tt)
#         ax.axis("off")
#     if title: plt.suptitle(title, fontsize=13)
#     plt.tight_layout(); plt.show()

# def enhance_uniform(img):    
#     img = to_gray01(img)
#     if USE_CLAHE: img = clahe01(img, clip=CLAHE_CLIP, tile=CLAHE_TILE)

#     mask  = tissue_mask(img)
#     inner = inner_mask_by_distance(mask, RIM_PX)
#     g     = grad_mean(img, K_GRAD)
#     g_masked = g * inner.astype(np.float32)

#     enhanced = enhance(img, g, inner, lam=LAMBDA, beta=BETA, sigma_detail=SIGMA_DETAIL)
#     return (enhanced*255).astype(np.uint8)




MAX_DIM     = 1600

USE_CLAHE   = True
CLAHE_CLIP  = 1.5
CLAHE_TILE  = 8

RIM_PX      = 20

# Nodulos (white top-hat multiescala)
BLOB_RADII        = [9, 13, 17]   # diametros
BLOB_SMOOTH_SIGMA = 1.2          # suavizado de la m�scara
BLOB_GAMMA        = 1.6          # >1 = m�s selectivo
BLOB_PCTL_CLIP    = 99.0         # recorte superior robusto

# Realce unsharp guiado por los nodulos 
SIGMA_DETAIL = 1.1               # radio del pasa-altos (? = menos grano)
LAMBDA       = 0.55              # fuerza del detalle
BETA         = 0.90              # mezcla con original (natural)
BASE_GAIN    = 0.15              # ?piso? de realce fuera de blobs (0?0.3)

def list_images(folder):
    paths = []
    for ext in EXTS: paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(list(dict.fromkeys(paths)))

def maybe_resize(img, max_dim):
    if max_dim is None: return img
    h,w = img.shape[:2]; m = max(h,w)
    if m <= max_dim: return img
    s = max_dim/float(m)
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def to_gray01(img):
    img = img.astype(np.float32)
    lo,hi = np.percentile(img,(0.5,99.5))
    return np.clip((img-lo)/max(1e-6,hi-lo),0,1)

def clahe01(img01, clip=1.5, tile=8):
    u8 = (np.clip(img01,0,1)*255).astype(np.uint8)
    out = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile,tile)).apply(u8)
    return out.astype(np.float32)/255.0

def tissue_mask(img01):
    blur = cv2.GaussianBlur(img01,(0,0),2.0)
    u8 = (blur*255).astype(np.uint8)
    thr,_ = cv2.threshold(u8,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    m = (u8>thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k)
    return m

def inner_mask_by_distance(mask, rim_px):
    if rim_px <= 0: return mask
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    inner = (dist > rim_px).astype(np.uint8)
    return (inner & mask).astype(np.uint8)

# ---------- M�scara de nodulos ----------
def blob_mask_whitetophat(img01):
    u8 = (np.clip(img01,0,1)*255).astype(np.uint8)
    acc = None
    for r in BLOB_RADII:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
        # top-hat blanco: resalta estructuras m�s peque�as que el kernel (redondeadas y brillantes)
        th = cv2.morphologyEx(u8, cv2.MORPH_TOPHAT, k).astype(np.float32)
        acc = th if acc is None else np.maximum(acc, th)
    # normaliza robusto
    p = np.percentile(acc, BLOB_PCTL_CLIP)
    acc = np.clip(acc, 0, max(p,1e-6)) / max(p,1e-6)
    # suaviza y aplica gamma (m�s selectivo)
    acc = cv2.GaussianBlur(acc, (0,0), BLOB_SMOOTH_SIGMA)
    acc = np.power(np.clip(acc,0,1), BLOB_GAMMA)
    # re-escala a [0,1]
    if acc.max() > 1e-6: acc /= acc.max()
    return acc.astype(np.float32)

def detail_layer(img01, sigma=1.1):
    blur = cv2.GaussianBlur(img01,(0,0),sigma)
    D = img01 - blur
    p = np.percentile(np.abs(D), 95.0) + 1e-6
    return D / p

def enhance_with_blobs(img01, blob01, mask_inner, lam=LAMBDA, beta=BETA, sigma_detail=SIGMA_DETAIL, base_gain=BASE_GAIN):
    Dn = detail_layer(img01, sigma_detail)
    G  = blob01 * mask_inner.astype(np.float32)
    gain = base_gain + (1.0 - base_gain) * G     
    enh = np.clip(img01 + lam * (gain * Dn), 0, 1)
    out = (1.0-beta)*img01 + beta*enh
    return out

def remove_black_specks_inpaint(img01, inner, win=9, tau=0.05,
                                dil=3, inpaint_radius=2, method="telea"):
    mu = cv2.boxFilter(img01, -1, (win, win), normalize=True)
    m = ((mu - img01) > tau).astype(np.uint8)
    m &= inner
    if dil and dil >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil))
        m = cv2.dilate(m, k)
    u8 = (np.clip(img01,0,1)*255).astype(np.uint8)
    mask_u8 = (m*255).astype(np.uint8)
    alg = cv2.INPAINT_TELEA if method.lower()=="telea" else cv2.INPAINT_NS
    out_u8 = cv2.inpaint(u8, mask_u8, inpaint_radius, alg)
    return out_u8.astype(np.float32)/255.0

def enhance_uniform(img):    
    img = to_gray01(img)
    if USE_CLAHE: img = clahe01(img, clip=CLAHE_CLIP, tile=CLAHE_TILE)

    mask  = tissue_mask(img)
    inner = inner_mask_by_distance(mask, RIM_PX)

    blob = blob_mask_whitetophat(img)
    blob_masked = np.clip(blob * inner.astype(np.float32), 0, 1)

    # enhanced = enhance_with_blobs(img, blob, inner)
    enhanced = img*(1-blob)
    final = remove_black_specks_inpaint(
        enhanced, inner, win=9, tau=0.05, dil=3, inpaint_radius=2, method="telea"
    )

    # k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    # enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_DILATE, k, iterations = 3) #cierre: rellena pequeños agujeros en la máscara


    return (final*255).astype(np.uint8)

# # ===== Ejecutar =====
# paths = list_images(INPUT_DIR)
# for p in paths:
#     raw = cv2.imread(p, cv2.IMREAD_UNCHANGED)
#     if raw is None: continue
#     raw = maybe_resize(raw, MAX_DIM)
#     img = to_gray01(raw)
#     if USE_CLAHE: img = clahe01(img, clip=CLAHE_CLIP, tile=CLAHE_TILE)

#     mask  = tissue_mask(img)
#     inner = inner_mask_by_distance(mask, RIM_PX)
#     g     = grad_mean(img, K_GRAD)
#     g_masked = g * inner.astype(np.float32)

#     enhanced = enhance(img, g, inner, lam=LAMBDA, beta=BETA, sigma_detail=SIGMA_DETAIL)

#     show_triplet(img, g_masked, enhanced, title=os.path.basename(p))



