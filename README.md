# Visão Computacional — Detecção Facial

Atividade da disciplina de **Visão Computacional**: detecção de rostos usando
**Haar Cascades** com OpenCV em Python.

## Conteúdo

| Arquivo | Descrição |
|---------|-----------|
| `deteccaoHaar.py` | Template original — detecção de rostos em uma **imagem** (`candidatos.jpg`). |
| `deteccaoVideo.py` | Detecção de rostos em **vídeo**, frame a frame (Haar puro). |
| `deteccaoVideoTracking.py` | Versão melhorada — **detecção + rastreamento** (Haar + tracker CSRT) que segue o rosto suavemente. |
| `haarcascade_frontalface_default.xml` | Modelo treinado para rostos frontais. |
| `frontalEyes35x16.xml`, `parojos.xml`, `parojosG.xml` | Modelos treinados para olhos. |
| `haarcascade_mcs_nose.xml` | Modelo treinado para nariz. |
| `Mouth.xml` | Modelo treinado para boca. |
| `candidatos.jpg` | Imagem de teste. |
| `videoplayback.mp4`, `falling_in_reverse.mp4` | Vídeos de teste. |
| `videoplayback_detectado.mp4` | Vídeo de saída com os rostos já marcados (prova de execução). |
| `Curso de Detecção Facial .pdf` | Apresentação da disciplina. |

## Como rodar

Requer Python 3 + OpenCV:

```bash
pip install opencv-python numpy
```

Detecção em imagem:

```bash
python deteccaoHaar.py
```

Detecção em vídeo (passe o arquivo como argumento; sem argumento usa `videoplayback.mp4`):

```bash
python deteccaoVideo.py falling_in_reverse.mp4
```

Detecção **com rastreamento** (caixa acompanha o rosto, sem piscar):

```bash
python deteccaoVideoTracking.py falling_in_reverse.mp4
```

> O `deteccaoVideoTracking.py` precisa do tracker CSRT, que vem no pacote
> `opencv-contrib-python` (em vez do `opencv-python`).

Aperte **`q`** para fechar a janela antes do fim do vídeo.

## Como funciona

O script percorre o vídeo quadro a quadro, converte cada frame para tons de
cinza e aplica o classificador Haar Cascade (`detectMultiScale`), desenhando um
retângulo em cada rosto detectado.

> **Observação:** o Haar Cascade faz detecção **independente em cada frame** (não
> é rastreamento). Por isso a caixa pode "piscar" quando o rosto vira, se move
> rápido ou a iluminação muda — comportamento esperado da técnica.
>
> O `deteccaoVideoTracking.py` resolve isso: o Haar detecta os rostos a cada 15
> frames e, no meio do caminho, um **tracker CSRT segue** cada rosto frame a
> frame. Um rosto só continua na tela enquanto for **reconfirmado** pelo Haar de
> tempos em tempos, o que elimina os falsos positivos do fundo.
