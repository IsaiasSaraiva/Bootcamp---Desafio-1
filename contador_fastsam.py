"""
Contador de parafusos com FastSAM (segmentacao "everything", sem treino).
Desafio 1 - Bootcamp CDIA / Residencia em IA.

Ideia: o FastSAM segmenta TODOS os objetos da cena sem precisar de dados rotulados.
Para contar parafusos, filtramos as mascaras retornadas (tira fundo, ruido, bordas e
mascaras duplicadas/aninhadas) e contamos quantas sobram.

A separacao de objetos do FastSAM e muito melhor que a morfologia para parafusos
encostados/amontoados - que era o ponto fraco da versao classica.

Obs.: os pesos (FastSAM-s.pt) sao baixados automaticamente na primeira execucao
(precisa de internet na primeira vez).
"""
import cv2
import numpy as np

# ---- knobs (ajuste se super/subcontar) -------------------------------------
MIN_AREA_FRAC = 0.0006   # mascaras menores que isso (do total) = ruido
MAX_AREA_FRAC = 0.30     # maiores que isso = fundo/mesa
DEDUP_IOU     = 0.80     # mascaras com IoU acima disso = mesma peca (mantem 1)
CONTAINED     = 0.75     # se uma mascara esta majoritariamente dentro de outra = aninhada
BORDER_REJECT = 2        # mascara encostando em >=2 bordas = mesa/recorte
MIN_SIDE_PX   = 14       # caixa com largura OU altura menor que isso = caco/ruido
TEXTURE_STD   = 12.0     # desvio-padrao de cinza dentro da mascara abaixo disso = fundo liso
MIN_CONF      = 0.40     # mascaras do FastSAM com confianca abaixo disso = descartadas
ENABLE_SPLIT  = True     # tenta re-separar mascara muito maior que a mediana (aglomerado)
SPLIT_RATIO   = 1.8      # area > SPLIT_RATIO * mediana -> tenta dividir
# ----------------------------------------------------------------------------

_MODEL = None


def _get_model(weights="FastSAM-s.pt"):
    """Carrega o FastSAM uma unica vez (lazy)."""
    global _MODEL
    if _MODEL is None:
        from ultralytics import FastSAM  # import tardio: app sobe mesmo sem o modelo
        _MODEL = FastSAM(weights)
    return _MODEL


def _split_mask(mask, k):
    """Tenta separar uma mascara que parece conter varias pecas (erosao -> watershed)."""
    comp = mask.astype(np.uint8) * 255
    seeds, best_n = comp.copy(), 1
    for it in range(1, 30):
        er = cv2.erode(comp, k, iterations=it)
        nn, _ = cv2.connectedComponents(er)
        if nn - 1 == 0:
            break
        seeds, best_n = er.copy(), nn - 1
    if best_n < 2:
        return [mask]
    _, mk = cv2.connectedComponents(seeds)
    mk = mk + 1
    mk[cv2.subtract(comp, seeds) > 0] = 0
    mk = cv2.watershed(cv2.cvtColor(comp, cv2.COLOR_GRAY2BGR), mk)
    parts = []
    for v in [u for u in np.unique(mk) if u > 1]:
        parts.append(mk == v)
    return parts or [mask]


def count_from_masks(masks, H, W, base_img=None, gray=None, scores=None):
    """
    Funcao PURA (testavel sem o modelo): recebe mascaras booleanas (HxW), opcionalmente
    a imagem em cinza (p/ filtro de textura) e as confiancas do FastSAM.
    Devolve (contagem, imagem_anotada|None, mascaras_mantidas).
    """
    A = H * W
    if scores is None:
        scores = [1.0] * len(masks)
    cands = []
    for m, sc in zip(masks, scores):
        if sc is not None and sc < MIN_CONF:                  # filtro de confianca
            continue
        m = np.asarray(m, dtype=bool)
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        area = int(m.sum())
        if area < MIN_AREA_FRAC * A or area > MAX_AREA_FRAC * A:
            continue
        ys, xs = np.where(m)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        if (x1 - x0) < MIN_SIDE_PX or (y1 - y0) < MIN_SIDE_PX:  # caco minusculo
            continue
        borders = int(x0 == 0) + int(y0 == 0) + int(x1 >= W - 1) + int(y1 >= H - 1)
        if borders >= BORDER_REJECT:
            continue
        if gray is not None and float(gray[m].std()) < TEXTURE_STD:  # fundo liso = sem textura
            continue
        cands.append({"mask": m, "area": area, "bbox": (x0, y0, x1, y1)})

    # dedup guloso: comeca pelas maiores; descarta a que repete (IoU alto) ou aninha
    cands.sort(key=lambda c: -c["area"])
    kept = []
    for c in cands:
        dup = False
        for k in kept:
            inter = int(np.logical_and(c["mask"], k["mask"]).sum())
            if inter == 0:
                continue
            union = int(np.logical_or(c["mask"], k["mask"]).sum())
            if union and inter / union > DEDUP_IOU:
                dup = True
                break
            if inter / c["area"] > CONTAINED:    # c quase contida em k
                dup = True
                break
        if not dup:
            kept.append(c)

    # re-split: mascara muito maior que a mediana provavelmente e um aglomerado
    if ENABLE_SPLIT and len(kept) >= 2:
        median = float(np.median([c["area"] for c in kept]))
        ksp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        expanded = []
        for c in kept:
            if c["area"] > SPLIT_RATIO * median:
                parts = _split_mask(c["mask"], ksp)
                if len(parts) >= 2:
                    for p in parts:
                        if int(p.sum()) < MIN_AREA_FRAC * A:
                            continue
                        ys, xs = np.where(p)
                        bb = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                        expanded.append({"mask": p, "area": int(p.sum()), "bbox": bb})
                    continue
            expanded.append(c)
        kept = expanded

    annotated = None
    if base_img is not None:
        annotated = base_img.copy()
        overlay = base_img.copy()
        rng = np.random.default_rng(0)
        for c in kept:
            color = tuple(int(v) for v in rng.integers(60, 255, 3))
            overlay[c["mask"]] = color
        annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)
        for i, c in enumerate(kept, 1):
            x0, y0, x1, y1 = c["bbox"]
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(annotated, str(i), ((x0 + x1) // 2 - 8, (y0 + y1) // 2 + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Total: {len(kept)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return len(kept), annotated, kept


def count_screws_sam(image, weights="FastSAM-s.pt",
                     conf=0.4, iou=0.9, imgsz=1024):
    """
    Aceita caminho (str) ou array BGR. Roda o FastSAM e conta os parafusos.
    Retorna (contagem:int, imagem_anotada:np.ndarray BGR).
    """
    img = cv2.imread(image) if isinstance(image, str) else image
    if img is None:
        raise FileNotFoundError(image)
    H, W = img.shape[:2]

    res = _get_model(weights).predict(
        img, retina_masks=True, conf=conf, iou=iou, imgsz=imgsz, verbose=False
    )
    r = res[0]
    if r.masks is None or r.masks.data is None or len(r.masks.data) == 0:
        out = img.copy()
        cv2.putText(out, "Total: 0", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return 0, out

    masks = r.masks.data.cpu().numpy().astype(bool)
    try:
        scores = r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None else None
    except Exception:
        scores = None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    count, annotated, _ = count_from_masks(masks, H, W, base_img=img, gray=gray, scores=scores)
    return count, annotated
