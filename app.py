
import base64
import csv
import glob
import io
import math
import os
import time
import traceback

import cv2
import numpy as np
from flask import (Flask, render_template, request, redirect, url_for,
                   Response, send_from_directory, abort)

from contador_fastsam import count_screws_sam

app = Flask(__name__)
ENGINE = "FastSAM-s"
WEIGHTS = "FastSAM-s.pt"
ASSETS_DIR = "Assets"
HISTORY = []  # {"pred", "truth", "err", "ms": latencia, "res": "WxH"}


def _logo_file():
    """Acha a primeira imagem na pasta Assets/ para exibir no topo (ou None)."""
    for f in sorted(glob.glob(os.path.join(ASSETS_DIR, "*"))):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".svg")):
            return os.path.basename(f)
    return None


def _model_size_mb():
    try:
        return round(os.path.getsize(WEIGHTS) / 1e6, 1)
    except OSError:
        return None


def _performance():
    """Metricas operacionais (nao precisam de gabarito): latencia, throughput, modelo."""
    lat = [h["ms"] for h in HISTORY if h.get("ms") is not None]
    if not lat:
        return {"n": 0, "engine": ENGINE, "model_mb": _model_size_mb()}
    avg = sum(lat) / len(lat)
    return {
        "n": len(lat),
        "engine": ENGINE,
        "model_mb": _model_size_mb(),
        "avg_ms": round(avg, 1),
        "min_ms": round(min(lat), 1),
        "max_ms": round(max(lat), 1),
        "fps": round(1000 / avg, 2) if avg > 0 else None,
    }


def _img_to_data_uri(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


def _aggregate():
    evald = [h for h in HISTORY if h["truth"] is not None]
    n = len(evald)
    if n == 0:
        return {"n": 0}
    errs = [h["pred"] - h["truth"] for h in evald]
    return {
        "n": n,
        "mae": round(sum(abs(e) for e in errs) / n, 3),
        "rmse": round(math.sqrt(sum(e * e for e in errs) / n), 3),
        "exact_acc": round(100 * sum(1 for e in errs if e == 0) / n, 1),
        "exact": sum(1 for e in errs if e == 0),
        "over": sum(1 for e in errs if e > 0),
        "under": sum(1 for e in errs if e < 0),
    }


def _page(**ctx):
    ctx.setdefault("result", None)
    ctx.setdefault("agg", _aggregate())
    ctx.setdefault("perf", _performance())
    ctx.setdefault("history", HISTORY[::-1])
    return render_template("index.html", logo=_logo_file(), **ctx)


@app.route("/", methods=["GET"])
def index():
    return _page()


@app.route("/assets/<path:fname>")
def assets(fname):
    if not os.path.isdir(ASSETS_DIR):
        abort(404)
    return send_from_directory(ASSETS_DIR, fname)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("imagem")
    if not file or file.filename == "":
        return redirect(url_for("index"))

    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return _page(result={"error": "Nao consegui ler a imagem enviada."})

    H, W = img.shape[:2]
    try:
        t0 = time.perf_counter()
        pred, annotated = count_screws_sam(img)
        latency_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        traceback.print_exc()
        msg = ("Falha ao rodar o FastSAM. Verifique se 'ultralytics' esta instalado e se "
               "houve internet para baixar os pesos na primeira execucao. Detalhe: " + str(e))
        return _page(result={"error": msg})

    truth_raw = (request.form.get("truth") or "").strip()
    truth = int(truth_raw) if truth_raw.isdigit() else None
    err = (pred - truth) if truth is not None else None
    HISTORY.append({"pred": pred, "truth": truth, "err": err,
                    "ms": round(latency_ms, 1), "res": f"{W}x{H}"})

    result = {
        "pred": pred, "truth": truth, "err": err,
        "abs_err": abs(err) if err is not None else None,
        "exact": (err == 0) if err is not None else None,
        "bias": (None if err is None else
                 "exato" if err == 0 else
                 "supercontou" if err > 0 else "subcontou"),
        "image_uri": _img_to_data_uri(annotated),
        "ms": round(latency_ms, 1), "res": f"{W}x{H}",
    }
    return _page(result=result)


@app.route("/exportar_csv")
def exportar_csv():
    """Exporta as metricas de avaliacao (resumo + detalhe por imagem) em CSV."""
    agg = _aggregate()
    perf = _performance()
    buf = io.StringIO()
    w = csv.writer(buf)

    w.writerow(["# RESUMO"])
    w.writerow(["metrica", "valor"])
    w.writerow(["motor", perf.get("engine")])
    w.writerow(["modelo_mb", perf.get("model_mb")])
    if agg.get("n"):
        w.writerow(["mae", agg["mae"]])
        w.writerow(["rmse", agg["rmse"]])
        w.writerow(["acuracia_exata_%", agg["exact_acc"]])
        w.writerow(["n_avaliadas", agg["n"]])
        w.writerow(["supercontagens", agg["over"]])
        w.writerow(["subcontagens", agg["under"]])
    if perf.get("n"):
        w.writerow(["latencia_media_ms", perf["avg_ms"]])
        w.writerow(["latencia_min_ms", perf["min_ms"]])
        w.writerow(["latencia_max_ms", perf["max_ms"]])
        w.writerow(["throughput_fps", perf["fps"]])
        w.writerow(["inferencias", perf["n"]])
    w.writerow([])

    w.writerow(["# DETALHE"])
    w.writerow(["indice", "previsto", "real", "erro", "erro_abs",
                "status", "latencia_ms", "resolucao"])
    for i, h in enumerate(HISTORY, 1):
        err = h.get("err")
        status = ("sem_gabarito" if err is None else
                  "exato" if err == 0 else
                  "super" if err > 0 else "sub")
        w.writerow([i, h["pred"],
                    h["truth"] if h["truth"] is not None else "",
                    err if err is not None else "",
                    abs(err) if err is not None else "",
                    status, h.get("ms", ""), h.get("res", "")])

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=metricas_picking.csv"},
    )


@app.route("/reset", methods=["POST"])
def reset():
    HISTORY.clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
