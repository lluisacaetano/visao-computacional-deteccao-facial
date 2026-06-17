import sys
import numpy as np
import cv2 as cv

# carrega xml de treinamento de identificacao de faces
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# permite passar o nome do video pela linha de comando.
# ex: python deteccaoVideo.py falling_in_reverse.mp4
# se nada for passado, usa o videoplayback.mp4 por padrao
arquivo = sys.argv[1] if len(sys.argv) > 1 else 'videoplayback.mp4'

# carrega o video
video = cv.VideoCapture(arquivo)

# loop que percorre cada frame (quadro) do video
while True:
    # le um frame do video: ret = True se ainda houver frames
    ret, frame = video.read()

    # se nao houver mais frames, encerra o loop (fim do video)
    if not ret:
        break

    # converte o frame para tons de cinza
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # detecta faces: scaleFactor = escala da imagem, minNeighbors = minimo de vizinhos
    # minSize ignora regioes muito pequenas (ruido)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # desenha um retangulo em cada rosto detectado
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # escreve na tela quantos rostos foram encontrados no frame atual
    cv.putText(frame, f'Rostos: {len(faces)}', (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # mostra o frame com os rostos marcados
    cv.imshow('Deteccao de Rostos - Video', frame)

    # aguarda 1ms por tecla; pressione 'q' para sair antes do fim
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# libera o video e fecha as janelas
video.release()
cv.destroyAllWindows()
