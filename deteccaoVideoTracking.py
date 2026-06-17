import sys
import numpy as np
import cv2 as cv

# ============================================================
#  DETECCAO + RASTREAMENTO de rostos em video
#
#  O Haar Cascade DETECTA os rostos de tempos em tempos.
#  Entre as deteccoes, um tracker CSRT SEGUE cada rosto frame
#  a frame -> a caixa acompanha o rosto de forma suave, em vez
#  de "piscar" como na deteccao pura.
# ============================================================

# carrega o xml de treinamento de rostos frontais
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# de quantos em quantos frames rodar a deteccao Haar de novo
# (para achar rostos novos e recuperar os que se perderam)
DETECT_INTERVAL = 15

# o quanto duas caixas precisam se sobrepor para serem o "mesmo rosto"
IOU_MINIMO = 0.3

# quantos ciclos de deteccao um tracker pode passar SEM ser reconfirmado
# pelo Haar antes de ser descartado (corta falsos positivos do fundo)
MAX_SEM_CONFIRMAR = 2


def iou(a, b):
    """Calcula a sobreposicao (Intersection over Union) entre duas caixas."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    uniao = aw * ah + bw * bh - inter
    return inter / uniao if uniao > 0 else 0


def detectar_rostos(frame):
    """Detecta rostos no frame com Haar Cascade."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # equalizeHist normaliza o brilho -> ajuda muito com iluminacao dificil
    gray = cv.equalizeHist(gray)
    # minNeighbors mais alto = deteccao mais rigorosa, menos falsos positivos
    return face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(40, 40))


# permite passar o nome do video pela linha de comando
arquivo = sys.argv[1] if len(sys.argv) > 1 else 'videoplayback.mp4'
video = cv.VideoCapture(arquivo)

# lista de rastreadores ativos:
# cada item = {'tracker': CSRT, 'box': (x,y,w,h), 'sem_confirmar': int}
rastreadores = []
n_frame = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    # ---- 1) ATUALIZA os rastreadores existentes (seguem o rosto) ----
    ainda_ativos = []
    for r in rastreadores:
        ok, box = r['tracker'].update(frame)
        if ok:
            r['box'] = tuple(int(v) for v in box)
            ainda_ativos.append(r)
        # se 'ok' for False, o tracker perdeu o rosto e e descartado
    rastreadores = ainda_ativos

    # ---- 2) DE TEMPOS EM TEMPOS, roda a deteccao Haar ----
    if n_frame % DETECT_INTERVAL == 0:
        deteccoes = detectar_rostos(frame)
        # todo tracker "envelhece" um ciclo; quem for reconfirmado volta a zero
        for r in rastreadores:
            r['sem_confirmar'] += 1
        for (x, y, w, h) in deteccoes:
            nova = (int(x), int(y), int(w), int(h))
            # esse rosto ja esta sendo seguido? se sim, apenas reconfirma
            seguido = False
            for r in rastreadores:
                if iou(nova, r['box']) > IOU_MINIMO:
                    r['sem_confirmar'] = 0
                    seguido = True
                    break
            # rosto novo -> cria um tracker para ele
            if not seguido:
                tracker = cv.TrackerCSRT_create()
                tracker.init(frame, nova)
                rastreadores.append({'tracker': tracker, 'box': nova, 'sem_confirmar': 0})
        # descarta trackers que ficaram ciclos demais sem o Haar reconfirmar
        rastreadores = [r for r in rastreadores if r['sem_confirmar'] <= MAX_SEM_CONFIRMAR]

    # ---- 3) DESENHA as caixas que estao sendo rastreadas ----
    for r in rastreadores:
        x, y, w, h = r['box']
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.putText(frame, 'Rastreando', (x, y - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv.putText(frame, f'Rostos: {len(rastreadores)}', (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow('Deteccao + Rastreamento de Rostos', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    n_frame += 1

video.release()
cv.destroyAllWindows()
