# Contador de Parafusos com FastSAM (Desafio 1 — CDIA)


## Como rodar como desenvolvimento

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
# 1a execução baixa os pesos FastSAM-s.pt (~23MB) automaticamente
# abrir http://127.0.0.1:5000
```

## Como rodar via docker


```bash
git clone
```
após clonar para máquina local descompate a pasta e execute os comandos via docker 

```bash
docker compose up -d
# abrir no navegador: http://127.0.0.1:5000
```



## Observações
- Métricas de **detecção** (Precisão/Recall/F1/mAP) exigem caixas de referência e são
  calculadas em avaliação **offline**, não nesta tela.
- FastSAM é zero-shot: ajuste os filtros no topo de `contador_fastsam.py` se super/subcontar.
