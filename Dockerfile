# Imagem base enxuta
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/app/.ultralytics

WORKDIR /app

# Dependencias de sistema necessarias para OpenCV e PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instala o PyTorch em versao CPU primeiro (evita baixar o build CUDA, que e enorme)
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

# Demais dependencias da aplicacao
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo da aplicacao
COPY . .

# Baixa os pesos do FastSAM ja no build, deixando o container self-contained.
# Se nao houver internet no build, os pesos serao baixados no 1o uso.
RUN python -c "from ultralytics import FastSAM; FastSAM('FastSAM-s.pt')" \
    || echo "Aviso: pesos nao baixados no build; serao obtidos na 1a inferencia."

EXPOSE 5000

# Servidor de producao.
# 1 worker: o estado (HISTORY) fica em memoria, entao varios workers nao o compartilhariam.
# threads para concorrencia leve; timeout alto porque a inferencia em CPU pode levar segundos.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "180", "app:app"]
