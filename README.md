# Picking · Contador de Parafusos com FastSAM (Desafio 1 — CDIA)

Interface web (Flask) que conta parafusos com **FastSAM** (segmentação sem treino),
com tema visual da **Residência IA**, **métricas de avaliação** acumuladas e
**exportação em CSV**.

## Como rodar

```bash
pip install -r requirements.txt
python app.py
# 1a execução baixa os pesos FastSAM-s.pt (~23MB) automaticamente
# abrir http://127.0.0.1:5000
```

## Recursos

- **Inferência**: sobe a imagem → FastSAM segmenta → filtramos (fundo, ruído, bordas,
  duplicatas, textura, confiança) e contamos → mostra contagem + imagem anotada.
- **Métricas de avaliação** (precisam da contagem real): MAE, RMSE, acurácia exata,
  viés super/sub, com tabela do histórico.
- **Desempenho operacional** (não precisa de gabarito): latência média, throughput
  (FPS), tamanho do modelo — atende ao critério de baixo recurso/celular.
- **Exportar CSV**: botão na seção de métricas (rota `/exportar_csv`). Traz uma seção
  **RESUMO** (todas as métricas) e uma seção **DETALHE** (uma linha por imagem:
  previsto, real, erro, erro_abs, status, latência, resolução).

## Logo / tema

- Coloque a logo na pasta `Assets/` (qualquer `.png/.jpg/.svg`); o app exibe a primeira
  imagem encontrada no topo, servida pela rota `/assets/<arquivo>`.
- As cores seguem a paleta da logo (azul-marinho + gradiente ciano → azul → roxo → magenta).

## Arquivos
- `app.py` — Flask (inferência, métricas, desempenho, export CSV, /assets).
- `contador_fastsam.py` — FastSAM + filtragem/contagem das máscaras.
- `templates/index.html` — interface (tema Residência IA).
- `Assets/` — logo exibida no topo.
- `requirements.txt` — dependências (CSV usa biblioteca padrão).

## Observações
- Métricas de **detecção** (Precisão/Recall/F1/mAP) exigem caixas de referência e são
  calculadas em avaliação **offline**, não nesta tela.
- FastSAM é zero-shot: ajuste os filtros no topo de `contador_fastsam.py` se super/subcontar.
